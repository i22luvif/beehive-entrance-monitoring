"""
Evalúa espacialmente el detector YOLO definitivo sobre la partición de test.

La evaluación se realiza mediante el validador de Ultralytics utilizando el
checkpoint indicado mediante --model y la partición test definida en data.yaml.

Esta evaluación es independiente de la comparación común realizada mediante
compare_detectors_local.py. En este script, Precision, Recall, mAP50 y mAP50-95
son las métricas devueltas directamente por Ultralytics.

Los resultados principales se guardan en:

    results/detection_evaluation/auditoria_deteccion_final.csv

Ultralytics genera además, dentro de spatial_evaluation/, las curvas,
matrices de confusión, visualizaciones y predicciones auxiliares.
"""


import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def evaluar_detector(ruta_modelo: Path, ruta_yaml: Path, resolucion: int, directorio_salida: Path) -> None:
    """Evalúa el detector YOLO sobre el conjunto de test y guarda las métricas espaciales."""

    if not ruta_modelo.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo}")

    if not ruta_yaml.is_file():
        raise FileNotFoundError(f"No se encontró data.yaml: {ruta_yaml}")

    # La carpeta se convierte a ruta absoluta antes de pasarla a Ultralytics.
    directorio_salida_absoluto = directorio_salida.resolve()
    directorio_salida_absoluto.mkdir(parents=True, exist_ok=True)

    print("Evaluación espacial del detector YOLO")
    print(f"Modelo: {ruta_modelo}")
    print(f"Dataset: {ruta_yaml}")
    print(f"Resolución: {resolucion}")
    print(f"Salida: {directorio_salida_absoluto}")

    # Se carga el checkpoint que se quiere evaluar.
    modelo = YOLO(str(ruta_modelo))

    # La validación se realiza sobre la partición test indicada en data.yaml.
    # No se fijan conf ni iou, por lo que se utilizan los valores del validador de Ultralytics.
    metricas = modelo.val(
        data=str(ruta_yaml),
        split="test",
        imgsz=resolucion,
        project=str(directorio_salida_absoluto),
        name="spatial_evaluation",
        save_json=True,
        plots=True,
    )

    # Se extraen las cuatro métricas principales devueltas por el evaluador.
    # map50 y map corresponden a mAP50 y mAP50-95.
    # mp y mr son la Precision y el Recall medios devueltos por Ultralytics.
    map50 = float(metricas.box.map50)
    map50_95 = float(metricas.box.map)
    precision = float(metricas.box.mp)
    recall = float(metricas.box.mr)

    # Las métricas principales se guardan también en un CSV para poder consultarlas directamente.
    resultados = pd.DataFrame({
        "Modelo": [str(ruta_modelo)],
        "mAP50": [map50],
        "mAP50_95": [map50_95],
        "Precision": [precision],
        "Recall": [recall],
    })

    ruta_csv = directorio_salida_absoluto / "auditoria_deteccion_final.csv"
    
    # Una nueva ejecución sobrescribe el CSV global existente dentro de la carpeta de salida.
    resultados.to_csv(ruta_csv, index=False)

    # Ultralytics guarda las curvas, matrices y predicciones dentro de esta subcarpeta.
    ruta_resultados_ultralytics = directorio_salida_absoluto / "spatial_evaluation"

    print("\nResultados finales:")
    print(f" - mAP50:     {map50:.2%}")
    print(f" - mAP50-95:  {map50_95:.2%}")
    print(f" - Precision: {precision:.2%}")
    print(f" - Recall:    {recall:.2%}")
    print(f"CSV guardado en: {ruta_csv}")
    print(f"Resultados de Ultralytics guardados en: {ruta_resultados_ultralytics}")


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Evalúa espacialmente el detector YOLO definitivo sobre el conjunto de test.")

    analizador.add_argument("--model", dest="ruta_modelo", type=Path, default=Path("model/best_yolo.pt"), help="Ruta al checkpoint YOLO que se va a evaluar.")
    analizador.add_argument("--data", dest="ruta_yaml", type=Path, default=Path("datasets/bee_dataset/data.yaml"), help="Ruta al archivo data.yaml del dataset.")
    analizador.add_argument("--imgsz", dest="resolucion", type=int, default=960, help="Resolución de entrada utilizada durante la evaluación.")
    analizador.add_argument("--output", dest="directorio_salida", type=Path, default=Path("results/detection_evaluation"), help="Carpeta donde se guardarán las métricas y gráficas.")

    argumentos = analizador.parse_args()

    # Se comprueban los parámetros básicos antes de comenzar la evaluación.
    if not argumentos.ruta_modelo.is_file():
        analizador.error(f"No existe el modelo: {argumentos.ruta_modelo}")

    if not argumentos.ruta_yaml.is_file():
        analizador.error(f"No existe el archivo data.yaml: {argumentos.ruta_yaml}")

    if argumentos.resolucion < 1:
        analizador.error("--imgsz debe ser un entero mayor o igual que 1.")

    return argumentos


def main() -> None:
    """Lee los argumentos y ejecuta la evaluación espacial del detector."""

    argumentos = obtener_argumentos()

    evaluar_detector(
        ruta_modelo=argumentos.ruta_modelo,
        ruta_yaml=argumentos.ruta_yaml,
        resolucion=argumentos.resolucion,
        directorio_salida=argumentos.directorio_salida,
    )


if __name__ == "__main__":
    main()