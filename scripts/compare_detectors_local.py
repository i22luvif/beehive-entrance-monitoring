"""
Compara YOLO26m, RT-DETR-L y Faster R-CNN sobre la misma partición de test.

Los tres detectores se transforman a una representación común de cajas xyxy.
La evaluación utiliza dos procedimientos:

- Precision y Recall:
  confianza mínima configurable, IoU >= 0.50 y emparejamiento uno a uno.

- mAP50, mAP50-95 y mAR100:
  cálculo común mediante TorchMetrics a partir de las predicciones conservadas.

También se mide el tiempo de procesamiento de cada detector. Las primeras imágenes
indicadas mediante --warmup-images participan en las métricas espaciales, pero
se excluyen de las estadísticas temporales.

Los resultados se almacenan en:

    results/detector_comparison/comparacion_detectores_test.csv
    results/detector_comparison/comparacion_detectores_test.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from tqdm import tqdm
from ultralytics import RTDETR, YOLO


EXTENSIONES_IMAGEN_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Faster R-CNN utiliza internamente la clase 0 para fondo y la clase 1 para abeja.
NUM_CLASES_FRCNN = 2

# Se mantiene un umbral muy bajo para conservar predicciones que se necesitan al calcular mAP y mAR.
UMBRAL_MINIMO_PREDICCIONES = 0.001

# IoU mínima utilizada en el cálculo común de Precision y Recall.
UMBRAL_IOU_PR = 0.50


@dataclass
class ResultadoDetector:
    """Guarda las métricas obtenidas por un detector sobre el conjunto de test."""

    modelo: str
    familia: str
    checkpoint: str
    numero_imagenes_test: int
    tp_iou50_conf: int
    fp_iou50_conf: int
    fn_iou50_conf: int
    precision_iou50_conf: float
    recall_iou50_conf: float
    map50: float
    map50_95: float
    mar100: float
    media_ms_imagen: float
    mediana_ms_imagen: float
    desviacion_ms_imagen: float
    fps: float
    dispositivo: str
    umbral_confianza: float
    imagenes_warmup_excluidas: int


def sincronizar_cuda(dispositivo: torch.device) -> None:
    """Sincroniza CUDA cuando la evaluación se está ejecutando en GPU."""

    if dispositivo.type == "cuda":
        torch.cuda.synchronize(dispositivo)


def calcular_media(valores: list[float]) -> float:
    """Devuelve la media de una lista o 0 si está vacía."""

    return float(np.mean(valores)) if valores else 0.0


def calcular_mediana(valores: list[float]) -> float:
    """Devuelve la mediana de una lista o 0 si está vacía."""

    return float(np.median(valores)) if valores else 0.0


def calcular_desviacion(valores: list[float]) -> float:
    """Devuelve la desviación típica poblacional o 0 si la lista está vacía."""

    return float(np.std(valores, ddof=0)) if valores else 0.0


def resolver_dispositivo(nombre_dispositivo: str) -> torch.device:
    """Devuelve el dispositivo solicitado y comprueba que CUDA esté disponible si se pide."""

    if nombre_dispositivo == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dispositivo = torch.device(nombre_dispositivo)

    if dispositivo.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se ha solicitado CUDA, pero PyTorch no detecta una GPU CUDA disponible.")

    return dispositivo


def dispositivo_para_ultralytics(dispositivo: torch.device) -> str | int:
    """Convierte torch.device al formato que espera Ultralytics."""

    if dispositivo.type == "cuda":
        return dispositivo.index if dispositivo.index is not None else 0

    return "cpu"


def resolver_rutas_dataset(ruta_yaml: Path, raiz_dataset_alternativa: Path | None) -> tuple[Path, Path]:
    """Obtiene las carpetas images/ y labels/ de la partición test."""

    if not ruta_yaml.exists():
        raise FileNotFoundError(f"No existe data.yaml: {ruta_yaml}")

    configuracion = yaml.safe_load(ruta_yaml.read_text(encoding="utf-8"))

    if not isinstance(configuracion, dict):
        raise ValueError("data.yaml no contiene un diccionario YAML válido.")

    # Si se proporciona --dataset-root, se utiliza esa carpeta en lugar del campo path de data.yaml.
    if raiz_dataset_alternativa is not None:
        directorio_raiz = raiz_dataset_alternativa.expanduser().resolve()
    else:
        directorio_raiz = Path(configuracion.get("path", ruta_yaml.parent)).expanduser()

        if not directorio_raiz.is_absolute():
            directorio_raiz = (ruta_yaml.parent / directorio_raiz).resolve()

    if not directorio_raiz.exists():
        raise FileNotFoundError(
            f"No existe el directorio raíz del dataset: {directorio_raiz}\n"
            "Usa --dataset-root si el campo 'path' de data.yaml apunta a otra máquina."
        )

    if "test" not in configuracion:
        raise ValueError("data.yaml no define el split 'test'.")

    directorio_imagenes_test = Path(str(configuracion["test"])).expanduser()

    if not directorio_imagenes_test.is_absolute():
        directorio_imagenes_test = (directorio_raiz / directorio_imagenes_test).resolve()

    directorio_etiquetas_test = directorio_imagenes_test.parent / "labels"

    if not directorio_imagenes_test.exists():
        raise FileNotFoundError(f"No existe la carpeta de imágenes TEST: {directorio_imagenes_test}")

    if not directorio_etiquetas_test.exists():
        raise FileNotFoundError(f"No existe la carpeta de etiquetas TEST: {directorio_etiquetas_test}")

    return directorio_imagenes_test, directorio_etiquetas_test


class DatasetTestYOLO(Dataset):
    """Lee el conjunto de test en formato YOLO y convierte las cajas a coordenadas xyxy."""

    def __init__(self, directorio_imagenes: Path, directorio_etiquetas: Path) -> None:
        self.directorio_imagenes = directorio_imagenes
        self.directorio_etiquetas = directorio_etiquetas

        # Se guardan ordenadas únicamente las imágenes con extensiones admitidas.
        self.imagenes = sorted(
            ruta for ruta in directorio_imagenes.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN_VALIDAS
        )

        if not self.imagenes:
            raise ValueError(f"No se encontraron imágenes en {directorio_imagenes}")

    def __len__(self) -> int:
        return len(self.imagenes)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], Path]:
        ruta_imagen = self.imagenes[indice]
        ruta_etiqueta = self.directorio_etiquetas / f"{ruta_imagen.stem}.txt"

        imagen = Image.open(ruta_imagen).convert("RGB")
        ancho_imagen, alto_imagen = imagen.size
        cajas: list[list[float]] = []

        if not ruta_etiqueta.exists():
            raise FileNotFoundError(f"No existe la etiqueta asociada a {ruta_imagen.name}: {ruta_etiqueta}")
        
        contenido = ruta_etiqueta.read_text(encoding="utf-8").strip()

        for linea in contenido.splitlines() if contenido else []:
            valores = linea.split()

            if len(valores) < 5:
                continue

            clase, centro_x, centro_y, ancho, alto = map(float, valores[:5])

            # El dataset tiene una única clase de objeto, identificada como 0 en formato YOLO.
            if int(clase) != 0:
                continue

            # Las cajas normalizadas se convierten a coordenadas absolutas xyxy.
            x1 = max(0.0, min(float(ancho_imagen), (centro_x - ancho / 2.0) * ancho_imagen))
            y1 = max(0.0, min(float(alto_imagen), (centro_y - alto / 2.0) * alto_imagen))
            x2 = max(0.0, min(float(ancho_imagen), (centro_x + ancho / 2.0) * ancho_imagen))
            y2 = max(0.0, min(float(alto_imagen), (centro_y + alto / 2.0) * alto_imagen))

            if x2 > x1 and y2 > y1:
                cajas.append([x1, y1, x2, y2])

        tensor_cajas = torch.tensor(cajas, dtype=torch.float32) if cajas else torch.zeros((0, 4), dtype=torch.float32)

        # Para la comparación común, la abeja se representa como clase 0 en TorchMetrics.
        anotacion = {
            "boxes": tensor_cajas,
            "labels": torch.zeros((tensor_cajas.shape[0],), dtype=torch.int64),
        }

        return TF.to_tensor(imagen), anotacion, ruta_imagen


def construir_fasterrcnn() -> torch.nn.Module:
    """Reconstruye Faster R-CNN con la misma arquitectura utilizada durante su entrenamiento."""

    # No se cargan de nuevo los pesos COCO porque el checkpoint entrenado sobrescribe los parámetros.
    modelo = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)

    # El predictor final trabaja con dos clases internas: fondo y abeja.
    numero_caracteristicas = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(numero_caracteristicas, NUM_CLASES_FRCNN)

    return modelo


def cargar_fasterrcnn(ruta_checkpoint: Path, dispositivo: torch.device) -> torch.nn.Module:
    """Carga el checkpoint de Faster R-CNN y deja el modelo preparado para inferencia."""

    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"No existe el checkpoint de Faster R-CNN: {ruta_checkpoint}")

    modelo = construir_fasterrcnn()

    # Se mantiene compatibilidad con versiones de PyTorch que no aceptan weights_only.
    try:
        checkpoint = torch.load(ruta_checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ruta_checkpoint, map_location="cpu")

    # Se aceptan checkpoints que guarden directamente el state_dict o lo incluyan dentro de un diccionario.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        estado_modelo = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        estado_modelo = checkpoint["state_dict"]
    else:
        estado_modelo = checkpoint

    if not isinstance(estado_modelo, dict):
        raise TypeError("El checkpoint de Faster R-CNN no contiene un state_dict reconocible.")

    # Se elimina el prefijo module. si el checkpoint procede de una ejecución con DataParallel.
    estado_modelo = {
        clave.replace("module.", "", 1) if clave.startswith("module.") else clave: valor
        for clave, valor in estado_modelo.items()
    }

    modelo.load_state_dict(estado_modelo, strict=True)

    # Se reduce el filtro interno para conservar predicciones de baja confianza necesarias para calcular AP y AR.
    modelo.roi_heads.score_thresh = UMBRAL_MINIMO_PREDICCIONES

    modelo.to(dispositivo)
    modelo.eval()

    return modelo


def calcular_pr_iou50(cajas_predichas: torch.Tensor, confianzas: torch.Tensor, cajas_reales: torch.Tensor, umbral_confianza: float) -> tuple[int, int, int]:
    """Calcula TP, FP y FN con emparejamiento uno a uno e IoU mínima de 0.50."""

    # Para Precision y Recall solo se utilizan predicciones que superan el umbral común de confianza.
    mascara_confianza = confianzas >= umbral_confianza
    cajas_predichas = cajas_predichas[mascara_confianza]
    confianzas = confianzas[mascara_confianza]

    if cajas_predichas.numel() == 0 and cajas_reales.numel() == 0:
        return 0, 0, 0

    if cajas_predichas.numel() == 0:
        return 0, 0, int(cajas_reales.shape[0])

    if cajas_reales.numel() == 0:
        return 0, int(cajas_predichas.shape[0]), 0

    # Las predicciones se procesan desde la confianza más alta a la más baja.
    orden = torch.argsort(confianzas, descending=True)
    cajas_predichas = cajas_predichas[orden]

    matriz_iou = box_iou(cajas_predichas, cajas_reales)
    indices_gt_usados: set[int] = set()
    verdaderos_positivos = 0
    falsos_positivos = 0

    for indice_prediccion in range(cajas_predichas.shape[0]):
        iou_disponibles = matriz_iou[indice_prediccion].clone()

        # Las anotaciones ya emparejadas se bloquean para que no puedan utilizarse de nuevo.
        if indices_gt_usados:
            indices_usados = torch.tensor(sorted(indices_gt_usados), dtype=torch.long)
            iou_disponibles[indices_usados] = -1.0

        mejor_iou, indice_mejor_gt = torch.max(iou_disponibles, dim=0)
        indice_gt = int(indice_mejor_gt.item())

        if float(mejor_iou.item()) >= UMBRAL_IOU_PR:
            verdaderos_positivos += 1
            indices_gt_usados.add(indice_gt)
        else:
            falsos_positivos += 1

    falsos_negativos = int(cajas_reales.shape[0] - len(indices_gt_usados))

    return verdaderos_positivos, falsos_positivos, falsos_negativos


def evaluar_detector(
    nombre_modelo: str,
    familia: str,
    funcion_inferencia: Callable[[torch.Tensor, np.ndarray], tuple[torch.Tensor, torch.Tensor]],
    dataset: DatasetTestYOLO,
    dispositivo: torch.device,
    umbral_confianza: float,
    imagenes_warmup: int,
    ruta_checkpoint: Path,
) -> ResultadoDetector:
    """Evalúa un detector con el mismo protocolo espacial y computacional que los demás modelos."""

    # TorchMetrics recibe las predicciones de los tres modelos con el mismo formato xyxy y la misma clase.
    evaluador_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    tiempos_ms: list[float] = []
    tp_total = 0
    fp_total = 0
    fn_total = 0

    print(f"\n{'=' * 72}\nEvaluando {nombre_modelo} ({familia})\nCheckpoint: {ruta_checkpoint}\n{'=' * 72}")

    with torch.inference_mode():
        for indice in tqdm(range(len(dataset)), desc=f"Test {nombre_modelo}", unit="img"):
            tensor_imagen, anotacion, _ = dataset[indice]

            # La lectura desde disco y esta conversión quedan fuera del cronómetro.
            imagen_rgb = tensor_imagen.permute(1, 2, 0).mul(255).byte().numpy()
            imagen_bgr = imagen_rgb[:, :, ::-1].copy()

            sincronizar_cuda(dispositivo)
            inicio = time.perf_counter()

            cajas_predichas, confianzas = funcion_inferencia(tensor_imagen, imagen_bgr)

            sincronizar_cuda(dispositivo)
            tiempo_ms = (time.perf_counter() - inicio) * 1000.0

            # Las imágenes iniciales participan en las métricas, pero no en el cálculo del tiempo.
            if indice >= imagenes_warmup:
                tiempos_ms.append(tiempo_ms)

            cajas_predichas = cajas_predichas.detach().cpu().float()
            confianzas = confianzas.detach().cpu().float()
            cajas_reales = anotacion["boxes"].detach().cpu().float()

            prediccion_torchmetrics = {
                "boxes": cajas_predichas,
                "scores": confianzas,
                "labels": torch.zeros((cajas_predichas.shape[0],), dtype=torch.int64),
            }

            referencia_torchmetrics = {
                "boxes": cajas_reales,
                "labels": torch.zeros((cajas_reales.shape[0],), dtype=torch.int64),
            }

            # mAP y mAR utilizan todas las predicciones conservadas y sus valores de confianza.
            evaluador_map.update([prediccion_torchmetrics], [referencia_torchmetrics])

            # Precision y Recall utilizan el umbral de confianza común y el emparejamiento uno a uno.
            tp, fp, fn = calcular_pr_iou50(cajas_predichas, confianzas, cajas_reales, umbral_confianza)

            tp_total += tp
            fp_total += fp
            fn_total += fn

    metricas_map = evaluador_map.compute()

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    media_ms = calcular_media(tiempos_ms)

    return ResultadoDetector(
        modelo=nombre_modelo,
        familia=familia,
        checkpoint=str(ruta_checkpoint),
        numero_imagenes_test=len(dataset),
        tp_iou50_conf=tp_total,
        fp_iou50_conf=fp_total,
        fn_iou50_conf=fn_total,
        precision_iou50_conf=precision,
        recall_iou50_conf=recall,
        map50=float(metricas_map["map_50"].item()),
        map50_95=float(metricas_map["map"].item()),
        mar100=float(metricas_map["mar_100"].item()),
        media_ms_imagen=media_ms,
        mediana_ms_imagen=calcular_mediana(tiempos_ms),
        desviacion_ms_imagen=calcular_desviacion(tiempos_ms),
        fps=(1000.0 / media_ms) if media_ms > 0 else 0.0,
        dispositivo=str(dispositivo),
        umbral_confianza=umbral_confianza,
        imagenes_warmup_excluidas=min(imagenes_warmup, len(dataset)),
    )


def convertir_resultado_a_fila(resultado: ResultadoDetector, epocas: int) -> dict:
    """Convierte el resultado interno a los nombres utilizados en los archivos CSV y JSON."""

    return {
        "model": resultado.modelo,
        "family": resultado.familia,
        "checkpoint": resultado.checkpoint,
        "num_test_images": resultado.numero_imagenes_test,
        "tp_iou50_conf": resultado.tp_iou50_conf,
        "fp_iou50_conf": resultado.fp_iou50_conf,
        "fn_iou50_conf": resultado.fn_iou50_conf,
        "precision_iou50_conf": resultado.precision_iou50_conf,
        "recall_iou50_conf": resultado.recall_iou50_conf,
        "map50": resultado.map50,
        "map50_95": resultado.map50_95,
        "mar100": resultado.mar100,
        "avg_ms_frame": resultado.media_ms_imagen,
        "median_ms_frame": resultado.mediana_ms_imagen,
        "std_ms_frame": resultado.desviacion_ms_imagen,
        "fps": resultado.fps,
        "device": resultado.dispositivo,
        "confidence_threshold": resultado.umbral_confianza,
        "warmup_images_excluded": resultado.imagenes_warmup_excluidas,
        "epochs": epocas,
    }


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Compara YOLO26m, RT-DETR-L y Faster R-CNN sobre el mismo conjunto de test.")

    analizador.add_argument("--data", dest="ruta_yaml", type=Path, required=True, help="Ruta al archivo data.yaml del dataset.")
    analizador.add_argument("--dataset-root", dest="raiz_dataset_alternativa", type=Path, default=None, help="Raíz local del dataset si el campo path de data.yaml apunta a otra máquina.")
    analizador.add_argument("--yolo", dest="ruta_yolo", type=Path, required=True, help="Checkpoint de YOLO utilizado en la comparación.")
    analizador.add_argument("--rtdetr", dest="ruta_rtdetr", type=Path, required=True, help="Checkpoint de RT-DETR utilizado en la comparación.")
    analizador.add_argument("--fasterrcnn", dest="ruta_fasterrcnn", type=Path, required=True, help="Checkpoint de Faster R-CNN utilizado en la comparación.")
    analizador.add_argument("--output", dest="directorio_salida", type=Path, default=Path("results/detector_comparison"), help="Carpeta donde se guardarán los resultados.")
    analizador.add_argument("--imgsz", dest="resolucion", type=int, default=960, help="Resolución utilizada por YOLO y RT-DETR durante la inferencia.")
    analizador.add_argument("--conf", dest="umbral_confianza", type=float, default=0.25, help="Umbral común de confianza para calcular Precision y Recall.")
    analizador.add_argument("--warmup-images", dest="imagenes_warmup", type=int, default=30, help="Imágenes iniciales excluidas únicamente de la medición temporal.")
    analizador.add_argument("--device", dest="nombre_dispositivo", default="auto", help="Dispositivo utilizado: auto, cpu, cuda:0, etc.")

    # Estos argumentos solo se guardan como información del entrenamiento de cada checkpoint.
    analizador.add_argument("--yolo-epochs", dest="epocas_yolo", type=int, default=15, help="Número de épocas del checkpoint YOLO.")
    analizador.add_argument("--rtdetr-epochs", dest="epocas_rtdetr", type=int, default=13, help="Número de épocas del checkpoint RT-DETR.")
    analizador.add_argument("--fasterrcnn-epochs", dest="epocas_fasterrcnn", type=int, default=16, help="Número de épocas del checkpoint Faster R-CNN.")

    argumentos = analizador.parse_args()

    # Se comprueban los parámetros principales antes de comenzar la evaluación.
    if not 0.0 <= argumentos.umbral_confianza <= 1.0:
        analizador.error("--conf debe estar entre 0 y 1.")

    if argumentos.imagenes_warmup < 0:
        analizador.error("--warmup-images no puede ser negativo.")

    if argumentos.resolucion <= 0:
        analizador.error("--imgsz debe ser mayor que 0.")
        
    if argumentos.epocas_yolo <= 0:
        analizador.error("--yolo-epochs debe ser mayor que 0.")

    if argumentos.epocas_rtdetr <= 0:
        analizador.error("--rtdetr-epochs debe ser mayor que 0.")

    if argumentos.epocas_fasterrcnn <= 0:
        analizador.error("--fasterrcnn-epochs debe ser mayor que 0.")

    return argumentos


def main() -> None:
    """Carga el dataset y los tres modelos, ejecuta la comparación y exporta los resultados."""

    argumentos = obtener_argumentos()

    # Se selecciona CPU o GPU según el argumento recibido.
    dispositivo = resolver_dispositivo(argumentos.nombre_dispositivo)
    dispositivo_ultralytics = dispositivo_para_ultralytics(dispositivo)

    print(f"Dispositivo: {dispositivo}")

    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(dispositivo)}")

    # Se cargan las imágenes y etiquetas correspondientes al mismo conjunto de test para los tres modelos.
    ruta_yaml = argumentos.ruta_yaml.expanduser().resolve()
    directorio_imagenes_test, directorio_etiquetas_test = resolver_rutas_dataset(
        ruta_yaml,
        argumentos.raiz_dataset_alternativa,
    )

    dataset = DatasetTestYOLO(directorio_imagenes_test, directorio_etiquetas_test)

    print(f"Imágenes TEST: {directorio_imagenes_test}")
    print(f"Etiquetas TEST: {directorio_etiquetas_test}")
    print(f"Número de imágenes TEST: {len(dataset)}")

    ruta_yolo = argumentos.ruta_yolo.expanduser().resolve()
    ruta_rtdetr = argumentos.ruta_rtdetr.expanduser().resolve()
    ruta_fasterrcnn = argumentos.ruta_fasterrcnn.expanduser().resolve()

    # Los tres checkpoints deben existir antes de comenzar la comparación.
    for ruta_checkpoint in [ruta_yolo, ruta_rtdetr, ruta_fasterrcnn]:
        if not ruta_checkpoint.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {ruta_checkpoint}")

    # YOLO utiliza el checkpoint correspondiente al entrenamiento comparativo.
    modelo_yolo = YOLO(str(ruta_yolo))

    def inferencia_yolo(tensor_imagen: torch.Tensor, imagen_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        del tensor_imagen

        # El umbral bajo permite conservar predicciones que después utiliza TorchMetrics.
        resultado = modelo_yolo(
            imagen_bgr,
            imgsz=argumentos.resolucion,
            conf=UMBRAL_MINIMO_PREDICCIONES,
            verbose=False,
            device=dispositivo_ultralytics,
        )[0]

        return resultado.boxes.xyxy, resultado.boxes.conf

    resultado_yolo = evaluar_detector(
        "YOLO26m",
        "Ultralytics YOLO",
        inferencia_yolo,
        dataset,
        dispositivo,
        argumentos.umbral_confianza,
        argumentos.imagenes_warmup,
        ruta_yolo,
    )

    # Se libera el modelo antes de cargar el siguiente detector.
    del modelo_yolo

    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()

    modelo_rtdetr = RTDETR(str(ruta_rtdetr))

    def inferencia_rtdetr(tensor_imagen: torch.Tensor, imagen_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        del tensor_imagen

        resultado = modelo_rtdetr(
            imagen_bgr,
            imgsz=argumentos.resolucion,
            conf=UMBRAL_MINIMO_PREDICCIONES,
            verbose=False,
            device=dispositivo_ultralytics,
        )[0]

        return resultado.boxes.xyxy, resultado.boxes.conf

    resultado_rtdetr = evaluar_detector(
        "RT-DETR-L",
        "Ultralytics RT-DETR",
        inferencia_rtdetr,
        dataset,
        dispositivo,
        argumentos.umbral_confianza,
        argumentos.imagenes_warmup,
        ruta_rtdetr,
    )

    del modelo_rtdetr

    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()

    # Faster R-CNN necesita reconstruir primero su arquitectura antes de cargar el state_dict.
    modelo_fasterrcnn = cargar_fasterrcnn(ruta_fasterrcnn, dispositivo)

    def inferencia_fasterrcnn(tensor_imagen: torch.Tensor, imagen_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        del imagen_bgr

        salida = modelo_fasterrcnn([tensor_imagen.to(dispositivo)])[0]

        # Solo existe una clase de objeto, por lo que se utilizan directamente las cajas y sus scores.
        return salida["boxes"], salida["scores"]

    resultado_fasterrcnn = evaluar_detector(
        "Faster R-CNN ResNet50-FPN v2",
        "Torchvision Faster R-CNN",
        inferencia_fasterrcnn,
        dataset,
        dispositivo,
        argumentos.umbral_confianza,
        argumentos.imagenes_warmup,
        ruta_fasterrcnn,
    )

    del modelo_fasterrcnn

    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()

    # Se mantienen los nombres de columnas y claves de los resultados existentes.
    filas = [
        convertir_resultado_a_fila(resultado_yolo, argumentos.epocas_yolo),
        convertir_resultado_a_fila(resultado_rtdetr, argumentos.epocas_rtdetr),
        convertir_resultado_a_fila(resultado_fasterrcnn, argumentos.epocas_fasterrcnn),
    ]

    tabla_resultados = pd.DataFrame(filas)

    columnas = [
        "model",
        "family",
        "epochs",
        "num_test_images",
        "tp_iou50_conf",
        "fp_iou50_conf",
        "fn_iou50_conf",
        "precision_iou50_conf",
        "recall_iou50_conf",
        "map50",
        "map50_95",
        "mar100",
        "avg_ms_frame",
        "median_ms_frame",
        "std_ms_frame",
        "fps",
        "confidence_threshold",
        "warmup_images_excluded",
        "device",
        "checkpoint",
    ]

    tabla_resultados = tabla_resultados[columnas]

    directorio_salida = argumentos.directorio_salida.expanduser().resolve()
    directorio_salida.mkdir(parents=True, exist_ok=True)

    ruta_csv = directorio_salida / "comparacion_detectores_test.csv"
    ruta_json = directorio_salida / "comparacion_detectores_test.json"

    # Se guardan tanto el CSV principal como una copia de las métricas en formato JSON.
    tabla_resultados.to_csv(ruta_csv, index=False)
    ruta_json.write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")

    # Esta copia se modifica solo para mostrar porcentajes y tiempos de forma más legible por consola.
    tabla_formateada = tabla_resultados.copy()

    for columna in ["precision_iou50_conf", "recall_iou50_conf", "map50", "map50_95", "mar100"]:
        tabla_formateada[columna] = tabla_formateada[columna].map(lambda valor: f"{valor:.2%}")

    for columna in ["avg_ms_frame", "median_ms_frame", "std_ms_frame", "fps"]:
        tabla_formateada[columna] = tabla_formateada[columna].map(lambda valor: f"{valor:.2f}")

    print("\n" + "=" * 120)
    print("COMPARACIÓN FINAL EN TEST")
    print("=" * 120)

    print(
        tabla_formateada[
            [
                "model",
                "epochs",
                "precision_iou50_conf",
                "recall_iou50_conf",
                "map50",
                "map50_95",
                "mar100",
                "avg_ms_frame",
                "fps",
            ]
        ].to_string(index=False)
    )

    print("=" * 120)
    print(f"\nCSV guardado en: {ruta_csv}")
    print(f"JSON guardado en: {ruta_json}")


if __name__ == "__main__":
    main()