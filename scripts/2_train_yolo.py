"""
Entrena YOLO26m para la detección de abejas utilizando Ultralytics.

El mismo script se utiliza en dos etapas distintas del proyecto:

1. Comparación de detectores:
   --epochs 15 --name yolo_15
   Genera model/best_yolo_15.pt.

2. Entrenamiento del detector definitivo:
   --epochs 150 --name yolo
   Genera model/best_yolo.pt.

El número indicado mediante --epochs es el máximo de épocas. El entrenamiento
puede finalizar antes mediante early stopping, con una paciencia de 25 épocas.

Ultralytics guarda los resultados en runs/train/<nombre>/ y el script copia
posteriormente el checkpoint best.pt a la carpeta model/.
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


# Parámetros de aumento de datos.
ROTACION_GRADOS = 180.0
PROB_MOSAICO = 1.0
PROB_MIXUP = 0.2
VARIACION_SATURACION = 0.3
VARIACION_BRILLO = 0.3

# Early stopping: número de épocas sin mejora antes de detener un entrenamiento que todavía no haya alcanzado su máximo.
EPOCAS_PACIENCIA = 25


def entrenar_yolo(ruta_yaml_datos: Path, ruta_pesos: Path, epocas: int, tamano_lote: int, resolucion: int, directorio_proyecto: Path, nombre_experimento: str) -> None:
    """Entrena YOLO26m y copia el mejor checkpoint a la carpeta model/."""

    if not ruta_yaml_datos.is_file():
        raise FileNotFoundError(f"No se encontró data.yaml: {ruta_yaml_datos}")

    # Se utiliza la primera GPU CUDA disponible y, si no hay ninguna, se ejecuta en CPU.
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    nombre_dispositivo = torch.cuda.get_device_name(0) if dispositivo == 0 else "CPU"

    print(f"Dispositivo: {nombre_dispositivo}")
    print(f"Cargando pesos iniciales: {ruta_pesos}")
    
    # Se carga YOLO a partir de los pesos preentrenados indicados.
    modelo = YOLO(str(ruta_pesos))

    # Ultralytics se encarga del entrenamiento, la validación y el guardado de los checkpoints.
    modelo.train(
        data=str(ruta_yaml_datos),
        epochs=epocas,
        patience=EPOCAS_PACIENCIA,
        imgsz=resolucion,
        batch=tamano_lote,
        device=dispositivo,
        project=str(directorio_proyecto),
        name=nombre_experimento,

        # Aumentos aplicados a las imágenes durante el entrenamiento.
        degrees=ROTACION_GRADOS,
        mosaic=PROB_MOSAICO,
        mixup=PROB_MIXUP,
        hsv_s=VARIACION_SATURACION,
        hsv_v=VARIACION_BRILLO,

        plots=True,
        exist_ok=True,
    )

    # Una vez terminado el entrenamiento, se localiza el mejor checkpoint guardado por Ultralytics.
    ruta_checkpoint = directorio_proyecto / nombre_experimento / "weights" / "best.pt"

    # El nombre del experimento determina también el checkpoint final:
    # yolo_15 -> best_yolo_15.pt; yolo -> best_yolo.pt.
    ruta_modelo_final = Path("model") / f"best_{nombre_experimento}.pt"
    ruta_modelo_final.parent.mkdir(parents=True, exist_ok=True)

    if not ruta_checkpoint.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint esperado: {ruta_checkpoint}")

    shutil.copy(ruta_checkpoint, ruta_modelo_final)
    print(f"Mejor checkpoint copiado a: {ruta_modelo_final}")


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Entrena YOLO26m para la detección de abejas.")

    analizador.add_argument("--data", dest="ruta_yaml_datos", type=Path, default=Path("datasets/bee_dataset/data.yaml"), help="Ruta al archivo data.yaml del dataset.")
    analizador.add_argument("--weights", dest="ruta_pesos", type=Path, default=Path("yolo26m.pt"), help="Pesos preentrenados utilizados para iniciar el entrenamiento.")
    analizador.add_argument("--epochs", dest="epocas", type=int, default=150, help="Número máximo de épocas de entrenamiento.")
    analizador.add_argument("--batch", dest="tamano_lote", type=int, default=8, help="Número de imágenes procesadas en cada lote.")
    analizador.add_argument("--imgsz", dest="resolucion", type=int, default=960, help="Resolución de entrada utilizada por el modelo.")
    analizador.add_argument("--project", dest="directorio_proyecto", type=Path, default=Path("runs/train"), help="Carpeta donde se guardan los resultados del entrenamiento.")
    analizador.add_argument("--name", dest="nombre_experimento", type=str, default="yolo", help="Nombre utilizado para identificar el experimento.")

    argumentos = analizador.parse_args()

    # Se comprueban los parámetros básicos antes de comenzar el entrenamiento.
    if argumentos.epocas < 1:
        analizador.error("--epochs debe ser un entero mayor o igual que 1.")

    if argumentos.tamano_lote < 1:
        analizador.error("--batch debe ser un entero mayor o igual que 1.")

    if argumentos.resolucion < 1:
        analizador.error("--imgsz debe ser un entero mayor o igual que 1.")

    if not argumentos.ruta_yaml_datos.is_file():
        analizador.error(f"No existe el archivo data.yaml: {argumentos.ruta_yaml_datos}")

    return argumentos


def main() -> None:
    """Lee los argumentos y ejecuta el entrenamiento de YOLO26m."""

    argumentos = obtener_argumentos()

    entrenar_yolo(
        ruta_yaml_datos=argumentos.ruta_yaml_datos,
        ruta_pesos=argumentos.ruta_pesos,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
        resolucion=argumentos.resolucion,
        directorio_proyecto=argumentos.directorio_proyecto,
        nombre_experimento=argumentos.nombre_experimento
    )


if __name__ == "__main__":
    main()