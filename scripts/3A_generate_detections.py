"""
Genera y almacena las detecciones del detector YOLO definitivo.

El detector procesa una única vez cada vídeo de evaluación temporal y guarda
sus predicciones en archivos CSV. Estos mismos archivos se utilizan después
como entrada común para todos los algoritmos de seguimiento, evitando ejecutar
YOLO de forma independiente para cada tracker.

Para cada vídeo se genera:

    detections_<video>.csv

con el formato:

    frame,x1,y1,x2,y2,conf,cls

También se generan detections_summary.csv y detection_run_config.json con las
medidas temporales y la configuración utilizada durante la ejecución.

Los primeros fotogramas indicados mediante --warmup_frames se procesan y se
guardan normalmente, pero se excluyen únicamente del cálculo de tiempos.
"""


import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
TAMANO_MINIMO_CAJA = 10

def sincronizar_cuda() -> None:
    """Sincroniza la GPU cuando CUDA está disponible."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def calcular_media(valores: list[float]) -> float:
    """Devuelve la media de una lista o 0 si está vacía."""
    return float(np.mean(valores)) if valores else 0.0


def calcular_mediana(valores: list[float]) -> float:
    """Devuelve la mediana de una lista o 0 si está vacía."""
    return float(np.median(valores)) if valores else 0.0


def calcular_desviacion(valores: list[float]) -> float:
    """Devuelve la desviación típica poblacional o 0 si la lista está vacía."""
    return float(np.std(valores, ddof=0)) if valores else 0.0


def generar_detecciones(ruta_modelo: Path, directorio_entrada: Path, directorio_salida: Path, umbral_confianza: float, fotogramas_warmup: int, resolucion: int) -> None:
    """Procesa los vídeos con YOLO y guarda sus detecciones y tiempos de ejecución."""

    # El dispositivo se selecciona automáticamente según la disponibilidad de CUDA.
    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Generación de detecciones almacenadas")
    print(f"Modelo: {ruta_modelo}")
    print(f"Entrada: {directorio_entrada}")
    print(f"Salida: {directorio_salida}")
    print(f"Dispositivo: {dispositivo}")

    # El detector se carga una sola vez y se reutiliza para todos los vídeos.
    modelo = YOLO(str(ruta_modelo))
    modelo.to(dispositivo)

    # Los archivos existentes con el mismo nombre se sobrescriben.
    # Para conservar una ejecución anterior debe utilizarse otra carpeta de salida.
    directorio_salida.mkdir(parents=True, exist_ok=True)

    # Se procesan únicamente los vídeos situados directamente en la carpeta de entrada, no se realiza una búsqueda recursiva.
    videos = sorted(ruta for ruta in directorio_entrada.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_VIDEO)

    if not videos:
        raise FileNotFoundError(f"No se encontraron vídeos compatibles en: {directorio_entrada}")

    resumen_videos = []

    for ruta_video in videos:
        nombre_video = ruta_video.stem
        ruta_csv_detecciones = directorio_salida / f"detections_{nombre_video}.csv"

        print(f"Procesando: {ruta_video.name}")

        captura_video = cv2.VideoCapture(str(ruta_video))

        if not captura_video.isOpened():
            raise RuntimeError(f"No se pudo abrir el vídeo: {ruta_video}")

        indice_fotograma = 0
        total_detecciones = 0

        # Aquí se guardan los tiempos de los fotogramas que entran en la medición.
        tiempos_deteccion_ms: list[float] = []

        # Cada vídeo genera su propio CSV con todas las detecciones encontradas.
        with ruta_csv_detecciones.open("w", newline="", encoding="utf-8") as archivo_csv:
            escritor = csv.writer(archivo_csv)

            # Cabecera del CSV: número de fotograma, caja, confianza y clase.
            escritor.writerow(["frame", "x1", "y1", "x2", "y2", "conf", "cls"])

            barra_progreso = tqdm(desc=f"Detectando {nombre_video}", unit="frame")

            # No se calculan gradientes porque el modelo se utiliza únicamente para inferencia.
            with torch.inference_mode():
                while captura_video.isOpened():
                    lectura_correcta, fotograma = captura_video.read()

                    if not lectura_correcta:
                        break

                    # Los fotogramas se numeran desde 1 para mantener la misma referencia en los CSV posteriores.
                    indice_fotograma += 1

                    # Los fotogramas iniciales de warm-up se procesan normalmente, pero no se incluyen en las medidas de tiempo.
                    medir_tiempo = indice_fotograma > fotogramas_warmup

                    # La medición incluye la inferencia, la transferencia de las predicciones a CPU y el filtro mínimo de tamaño. La escritura del CSV queda fuera.
                    if medir_tiempo:
                        sincronizar_cuda()
                        tiempo_inicio = time.perf_counter()

                    # Se ejecuta el detector con la confianza y resolución indicadas.
                    resultados = modelo(fotograma, verbose=False, conf=umbral_confianza, imgsz=resolucion)[0]

                    # Se pasan las detecciones a CPU y se descartan las cajas demasiado pequeñas.
                    cajas_validas = np.empty((0, 6))

                    if len(resultados.boxes) > 0:
                        cajas_brutas = np.hstack([
                            resultados.boxes.xyxy.cpu().numpy(),
                            resultados.boxes.conf.cpu().numpy().reshape(-1, 1),
                            resultados.boxes.cls.cpu().numpy().reshape(-1, 1),
                        ])

                        # La caja debe tener al menos el tamaño mínimo tanto en anchura como en altura.
                        filtro_tamano = (
                            (cajas_brutas[:, 2] - cajas_brutas[:, 0] >= TAMANO_MINIMO_CAJA)
                            & (cajas_brutas[:, 3] - cajas_brutas[:, 1] >= TAMANO_MINIMO_CAJA)
                        )

                        cajas_validas = cajas_brutas[filtro_tamano]

                    # El cronómetro termina antes de escribir las detecciones en el CSV.
                    if medir_tiempo:
                        sincronizar_cuda()
                        tiempo_fin = time.perf_counter()
                        tiempos_deteccion_ms.append((tiempo_fin - tiempo_inicio) * 1000)

                    # Cada fila guarda el fotograma, la caja, la confianza y la clase detectada.
                    for caja in cajas_validas:
                        x1, y1, x2, y2, confianza, clase = caja
                        escritor.writerow([indice_fotograma, x1, y1, x2, y2, confianza, int(clase)])
                        total_detecciones += 1

                    barra_progreso.update(1)

            barra_progreso.close()

        captura_video.release()

        # Se calculan las medidas de tiempo utilizando solo los fotogramas posteriores al warm-up.
        tiempo_medio_ms = calcular_media(tiempos_deteccion_ms)
        tiempo_mediano_ms = calcular_mediana(tiempos_deteccion_ms)
        desviacion_tiempo_ms = calcular_desviacion(tiempos_deteccion_ms)
        fps_deteccion = 1000.0 / tiempo_medio_ms if tiempo_medio_ms > 0 else 0.0

        print(f"Finalizado {nombre_video}: {indice_fotograma} fotogramas, {total_detecciones} detecciones, {tiempo_medio_ms:.1f} ms/fotograma, {fps_deteccion:.2f} FPS")

        resumen_videos.append([
            nombre_video,
            indice_fotograma,
            total_detecciones,
            tiempo_medio_ms,
            tiempo_mediano_ms,
            desviacion_tiempo_ms,
            fps_deteccion,
            str(ruta_csv_detecciones),
        ])

    # Se guarda un resumen con los conteos y las medidas de tiempo obtenidas para cada vídeo.
    ruta_resumen = directorio_salida / "detections_summary.csv"

    with ruta_resumen.open("w", newline="", encoding="utf-8") as archivo_resumen:
        escritor = csv.writer(archivo_resumen)
        escritor.writerow(["video", "num_frames", "num_detections", "avg_det_ms", "median_det_ms", "std_det_ms", "det_fps", "detections_csv"])
        escritor.writerows(resumen_videos)

    # La configuración se guarda junto a los CSV para saber con qué parámetros se generaron las detecciones.
    configuracion_ejecucion = {
        "model": str(ruta_modelo),
        "imgsz": resolucion,
        "confidence": umbral_confianza,
        "minimum_box_size_px": TAMANO_MINIMO_CAJA,
        "warmup_frames": fotogramas_warmup,
        "device": dispositivo,
    }

    ruta_configuracion = directorio_salida / "detection_run_config.json"

    with ruta_configuracion.open("w", encoding="utf-8") as archivo_configuracion:
        json.dump(configuracion_ejecucion, archivo_configuracion, indent=2, ensure_ascii=False)

    print(f"Resumen guardado en: {ruta_resumen}")
    print(f"Configuración guardada en: {ruta_configuracion}")
    print(f"Vídeos procesados: {len(resumen_videos)}")


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Genera y almacena las detecciones de YOLO para los vídeos de evaluación temporal.")

    analizador.add_argument("--model", dest="ruta_modelo", type=Path, default=Path("model/best_yolo.pt"), help="Ruta al detector YOLO utilizado para generar las detecciones.")
    analizador.add_argument("--input", dest="directorio_entrada", type=Path, default=Path("data/videos_evaluacion_temporal"), help="Carpeta que contiene los vídeos de evaluación temporal.")
    analizador.add_argument("--output", dest="directorio_salida", type=Path, default=Path("results/detections"), help="Carpeta donde se guardarán los CSV de detecciones.")
    analizador.add_argument("--conf", dest="umbral_confianza", type=float, default=0.25, help="Umbral mínimo de confianza utilizado por YOLO.")
    analizador.add_argument("--warmup_frames", dest="fotogramas_warmup", type=int, default=30, help="Número de fotogramas iniciales excluidos de las medidas de tiempo.")
    analizador.add_argument("--imgsz", dest="resolucion", type=int, default=960, help="Resolución de entrada utilizada durante la inferencia.")

    argumentos = analizador.parse_args()

    # Se comprueban los parámetros básicos antes de comenzar a procesar los vídeos.
    if not argumentos.directorio_entrada.is_dir():
        analizador.error(f"No existe la carpeta de entrada: {argumentos.directorio_entrada}")

    if not 0.0 <= argumentos.umbral_confianza <= 1.0:
        analizador.error("--conf debe estar entre 0 y 1.")

    if argumentos.fotogramas_warmup < 0:
        analizador.error("--warmup_frames debe ser mayor o igual que 0.")

    if argumentos.resolucion < 1:
        analizador.error("--imgsz debe ser un entero mayor o igual que 1.")
        
    if not argumentos.ruta_modelo.is_file():
        analizador.error(f"No existe el modelo indicado: {argumentos.ruta_modelo}")

    return argumentos


def main() -> None:
    """Lee los argumentos y genera las detecciones de todos los vídeos."""

    argumentos = obtener_argumentos()

    generar_detecciones(
        ruta_modelo=argumentos.ruta_modelo,
        directorio_entrada=argumentos.directorio_entrada,
        directorio_salida=argumentos.directorio_salida,
        umbral_confianza=argumentos.umbral_confianza,
        fotogramas_warmup=argumentos.fotogramas_warmup,
        resolucion=argumentos.resolucion,
    )


if __name__ == "__main__":
    main()