"""
Entrena Faster R-CNN ResNet50-FPN v2 para la detección de abejas.

El script utiliza el mismo dataset YOLO empleado por los demás detectores.
La clase ConjuntoDatosAbejas convierte internamente las etiquetas YOLO
normalizadas al formato de cajas absolutas requerido por Torchvision.

El modelo parte de pesos preentrenados y sustituye su predictor final para
trabajar con dos clases internas:

    0 = fondo
    1 = abeja

Durante cada época se calcula la pérdida de entrenamiento y la pérdida de
validación. El checkpoint con menor pérdida de validación se guarda en:

    runs/train/<nombre_experimento>/weights/best_fasterrcnn.pt

y posteriormente se copia a:

    model/best_<nombre_experimento>.pt

Con --name fasterrcnn se genera model/best_fasterrcnn.pt.
"""


import argparse
import shutil
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


EXTENSIONES_IMAGEN_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ConjuntoDatosAbejas(Dataset):
    """Lee el dataset YOLO y adapta sus anotaciones al formato usado por Faster R-CNN."""

    def __init__(self, directorio_imagenes: Path, directorio_etiquetas: Path) -> None:
        self.directorio_imagenes = Path(directorio_imagenes)
        self.directorio_etiquetas = Path(directorio_etiquetas)
        
        if not self.directorio_imagenes.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de imágenes: {self.directorio_imagenes}")

        if not self.directorio_etiquetas.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de etiquetas: {self.directorio_etiquetas}")

        # Se guardan ordenadas únicamente las imágenes con extensiones admitidas.
        self.rutas_imagenes = sorted(ruta for ruta in self.directorio_imagenes.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN_VALIDAS)

    def __len__(self) -> int:
        return len(self.rutas_imagenes)

    def __getitem__(self, indice: int) -> tuple:
        ruta_imagen = self.rutas_imagenes[indice]
        ruta_etiqueta = self.directorio_etiquetas / f"{ruta_imagen.stem}.txt"

        # La imagen se convierte a RGB para trabajar siempre con tres canales,
        # aunque el archivo original tenga otro modo de color.
        imagen = Image.open(ruta_imagen).convert("RGB")
        ancho_imagen, alto_imagen = imagen.size

        cajas_absolutas: list[list[float]] = []
        etiquetas_clase: list[int] = []

        if not ruta_etiqueta.is_file():
            raise FileNotFoundError(f"No existe la etiqueta asociada a {ruta_imagen.name}: {ruta_etiqueta}"
                                    )
        contenido_etiqueta = ruta_etiqueta.read_text(encoding="utf-8").strip() if ruta_etiqueta.exists() else ""

        if contenido_etiqueta:
            for linea in contenido_etiqueta.splitlines():
                id_clase, centro_x_norm, centro_y_norm, ancho_norm, alto_norm = map(float, linea.split()[:5])

                # En las etiquetas YOLO la abeja es la clase 0, pero Torchvision reserva el 0 para el fondo.
                if int(id_clase) != 0:
                    continue

                # Las coordenadas normalizadas de YOLO se convierten a coordenadas absolutas x1, y1, x2, y2.
                x1 = (centro_x_norm - ancho_norm / 2) * ancho_imagen
                y1 = (centro_y_norm - alto_norm / 2) * alto_imagen
                x2 = (centro_x_norm + ancho_norm / 2) * ancho_imagen
                y2 = (centro_y_norm + alto_norm / 2) * alto_imagen

                # Solo se guardan cajas con anchura y altura positivas.
                if x2 > x1 and y2 > y1:
                    cajas_absolutas.append([x1, y1, x2, y2])
                    etiquetas_clase.append(1)

        # Las imágenes de fondo se representan con un tensor de cajas vacío.
        if cajas_absolutas:
            tensor_cajas = torch.tensor(cajas_absolutas, dtype=torch.float32)
        else:
            tensor_cajas = torch.zeros((0, 4), dtype=torch.float32)

        tensor_etiquetas = torch.tensor(etiquetas_clase, dtype=torch.int64)

        # Estos nombres son los que espera la interfaz de detección de Torchvision.
        anotacion = {
            "boxes": tensor_cajas,
            "labels": tensor_etiquetas,
            "image_id": torch.tensor([indice]),
        }

        return F.to_tensor(imagen), anotacion


def agrupar_muestras_en_lote(lote: list) -> tuple:
    """Separa las imágenes y sus anotaciones para formar el lote que recibe Faster R-CNN."""
    return tuple(zip(*lote))


def cargar_rutas_desde_yaml(ruta_data_yaml: Path) -> dict[str, Path]:
    """Lee data.yaml y obtiene las rutas de imágenes y etiquetas de train y val."""

    ruta_yaml = Path(ruta_data_yaml)
    configuracion_yaml = yaml.safe_load(ruta_yaml.read_text(encoding="utf-8"))

    # La ruta raíz puede estar guardada como absoluta o relativa dentro de data.yaml.
    directorio_raiz = Path(configuracion_yaml.get("path", ruta_yaml.parent))

    if not directorio_raiz.is_absolute():
        directorio_raiz = (ruta_yaml.parent / directorio_raiz).resolve()

    directorio_imagenes_entrenamiento = directorio_raiz / configuracion_yaml["train"]
    directorio_imagenes_validacion = directorio_raiz / configuracion_yaml["val"]

    # Las carpetas labels/ se encuentran al mismo nivel que images/ dentro de cada partición.
    return {
        "imagenes_entrenamiento": directorio_imagenes_entrenamiento,
        "etiquetas_entrenamiento": directorio_imagenes_entrenamiento.parent / "labels",
        "imagenes_validacion": directorio_imagenes_validacion,
        "etiquetas_validacion": directorio_imagenes_validacion.parent / "labels",
    }


def construir_modelo_fasterrcnn() -> torch.nn.Module:
    """Carga Faster R-CNN ResNet50-FPN v2 preentrenado y adapta su predictor a fondo y abeja."""
    # Se parte de los pesos preentrenados por defecto de Torchvision.
    modelo = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # El predictor original se sustituye por uno de dos clases: 0 = fondo y 1 = abeja.
    numero_caracteristicas_entrada = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(numero_caracteristicas_entrada, 2)

    return modelo


def entrenar_fasterrcnn(ruta_data_yaml: Path, epocas: int, tamano_lote: int, tasa_aprendizaje: float, directorio_proyecto: Path, nombre_experimento: str) -> None:
    """Entrena Faster R-CNN y conserva el checkpoint con menor pérdida de validación."""

    rutas_dataset = cargar_rutas_desde_yaml(ruta_data_yaml)

    conjunto_entrenamiento = ConjuntoDatosAbejas(rutas_dataset["imagenes_entrenamiento"], rutas_dataset["etiquetas_entrenamiento"])
    conjunto_validacion = ConjuntoDatosAbejas(rutas_dataset["imagenes_validacion"], rutas_dataset["etiquetas_validacion"])
    
    if len(conjunto_entrenamiento) == 0:
        raise ValueError("El conjunto de entrenamiento no contiene imágenes.")

    if len(conjunto_validacion) == 0:
        raise ValueError("El conjunto de validación no contiene imágenes.")

    # El collate personalizado permite trabajar con un número distinto de cajas en cada imagen.
    cargador_entrenamiento = DataLoader(conjunto_entrenamiento, batch_size=tamano_lote, shuffle=True, collate_fn=agrupar_muestras_en_lote, num_workers=2)
    cargador_validacion = DataLoader(conjunto_validacion, batch_size=tamano_lote, shuffle=False, collate_fn=agrupar_muestras_en_lote, num_workers=2)

    # Se utiliza CUDA si está disponible y, en caso contrario, CPU.
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nombre_dispositivo = torch.cuda.get_device_name(0) if dispositivo.type == "cuda" else "CPU"

    print(f"[DISPOSITIVO] {nombre_dispositivo}")
    print(f"[DATOS] entrenamiento={len(conjunto_entrenamiento)} | validación={len(conjunto_validacion)}")

    modelo = construir_modelo_fasterrcnn().to(dispositivo)

    # Solo se optimizan los parámetros que tienen activado el cálculo de gradientes.
    optimizador = torch.optim.SGD([parametro for parametro in modelo.parameters() if parametro.requires_grad], lr=tasa_aprendizaje, momentum=0.9, weight_decay=0.0005)

    directorio_pesos_experimento = Path(directorio_proyecto) / nombre_experimento / "weights"
    directorio_pesos_experimento.mkdir(parents=True, exist_ok=True)

    mejor_perdida_validacion = float("inf")
    ruta_mejor_checkpoint = directorio_pesos_experimento / "best_fasterrcnn.pt"

    for numero_epoca in range(1, epocas + 1):
        modelo.train()
        perdida_entrenamiento_acumulada = 0.0

        for imagenes_lote, anotaciones_lote in cargador_entrenamiento:
            imagenes_dispositivo = [imagen.to(dispositivo) for imagen in imagenes_lote]
            anotaciones_dispositivo = [{clave: valor.to(dispositivo) for clave, valor in anotacion.items()} for anotacion in anotaciones_lote]

            # El modelo devuelve por separado las distintas pérdidas utilizadas por Faster R-CNN.
            perdidas_por_tipo = modelo(imagenes_dispositivo, anotaciones_dispositivo)
            perdida_total = sum(perdidas_por_tipo.values())

            # Se limpian los gradientes anteriores, se calculan los nuevos y se actualizan los pesos.
            optimizador.zero_grad()
            perdida_total.backward()
            optimizador.step()

            perdida_entrenamiento_acumulada += perdida_total.item()

        perdida_entrenamiento_media = perdida_entrenamiento_acumulada / max(len(cargador_entrenamiento), 1)

        # Faster R-CNN devuelve el diccionario de pérdidas cuando está en modo train().
        # Durante la validación se mantiene train() para obtener estas pérdidas.
        # torch.no_grad() evita calcular gradientes y no se ejecuta el optimizador.
        modelo.train()
        perdida_validacion_acumulada = 0.0

        with torch.no_grad():
            for imagenes_lote, anotaciones_lote in cargador_validacion:
                imagenes_dispositivo = [imagen.to(dispositivo) for imagen in imagenes_lote]
                anotaciones_dispositivo = [{clave: valor.to(dispositivo) for clave, valor in anotacion.items()} for anotacion in anotaciones_lote]

                perdidas_por_tipo = modelo(imagenes_dispositivo, anotaciones_dispositivo)
                perdida_validacion_acumulada += sum(perdidas_por_tipo.values()).item()

        perdida_validacion_media = perdida_validacion_acumulada / max(len(cargador_validacion), 1)

        print(f"[Época {numero_epoca:03d}] pérdida_train={perdida_entrenamiento_media:.4f} | pérdida_val={perdida_validacion_media:.4f}")

        # Cada vez que mejora la pérdida de validación se sustituye el mejor checkpoint anterior.
        if perdida_validacion_media < mejor_perdida_validacion:
            mejor_perdida_validacion = perdida_validacion_media
            torch.save(modelo.state_dict(), ruta_mejor_checkpoint)
            print(f"[OK] Nuevo mejor checkpoint guardado: {ruta_mejor_checkpoint}")

    # El mejor checkpoint se copia a model/ para utilizarlo después en la comparación de detectores.
    ruta_modelo_final = Path("model") / f"best_{nombre_experimento}.pt"
    ruta_modelo_final.parent.mkdir(parents=True, exist_ok=True)

    if ruta_mejor_checkpoint.exists():
        shutil.copy(ruta_mejor_checkpoint, ruta_modelo_final)
        print(f"[OK] Mejor checkpoint copiado a: {ruta_modelo_final}")


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Entrena Faster R-CNN ResNet50-FPN v2 para la detección de abejas.")

    analizador.add_argument("--data", dest="ruta_data_yaml", type=Path, default=Path("datasets/bee_dataset/data.yaml"), help="Ruta al archivo data.yaml del dataset.")
    analizador.add_argument("--epochs", dest="epocas", type=int, default=150, help="Número de épocas de entrenamiento.")
    analizador.add_argument("--batch", dest="tamano_lote", type=int, default=2, help="Número de imágenes procesadas en cada lote.")
    analizador.add_argument("--lr", dest="tasa_aprendizaje", type=float, default=0.0025, help="Tasa de aprendizaje utilizada por el optimizador SGD.")
    analizador.add_argument("--project", dest="directorio_proyecto", type=Path, default=Path("runs/train"), help="Carpeta donde se guardan los resultados del entrenamiento.")
    analizador.add_argument("--name", dest="nombre_experimento", type=str, default="fasterrcnn", help="Nombre utilizado para identificar el experimento.")

    argumentos = analizador.parse_args()

    # Se comprueban los parámetros básicos antes de comenzar el entrenamiento.
    if argumentos.epocas < 1:
        analizador.error("--epochs debe ser un entero mayor o igual que 1.")

    if argumentos.tamano_lote < 1:
        analizador.error("--batch debe ser un entero mayor o igual que 1.")

    if argumentos.tasa_aprendizaje <= 0:
        analizador.error("--lr debe ser mayor que 0.")

    if not argumentos.ruta_data_yaml.is_file():
        analizador.error(f"No existe el archivo data.yaml: {argumentos.ruta_data_yaml}")

    return argumentos


def main() -> None:
    """Lee los argumentos y ejecuta el entrenamiento de Faster R-CNN."""

    argumentos = obtener_argumentos()

    entrenar_fasterrcnn(
        ruta_data_yaml=argumentos.ruta_data_yaml,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
        tasa_aprendizaje=argumentos.tasa_aprendizaje,
        directorio_proyecto=argumentos.directorio_proyecto,
        nombre_experimento=argumentos.nombre_experimento
    )


if __name__ == "__main__":
    main()