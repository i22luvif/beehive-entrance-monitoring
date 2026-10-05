
"""
Compara varios algoritmos de seguimiento utilizando detecciones almacenadas.

Cada tracker recibe exactamente los mismos vídeos y los mismos CSV generados
por 3A_generate_detections.py. El detector YOLO no se ejecuta en esta etapa.

Para cada identificador devuelto por el tracker se mantiene un historial con:

    (fotograma, centroide, zona)

Cuando una trayectoria lleva 30 fotogramas sin aparecer, se analiza su historial
para generar eventos IN y OUT. Las trayectorias que permanecen abiertas al final
del vídeo también se procesan.

El script evalúa por defecto:

    ByteTrack
    BoT-SORT
    OC-SORT
    BoostTrack
    StrongSORT

Para cada combinación de tracker y vídeo se genera:

    results_<tracker>_<video>.csv

con el formato:

    frame,event

También se generan resúmenes de conteo y del tiempo empleado exclusivamente
por tracker.update(...).
"""

import argparse
import csv
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from boxmot import create_tracker
from tqdm import tqdm


# Geometría utilizada para clasificar la posición de cada trayectoria.
CAJA_CONTEO = (40, 560, 1210, 635)
LIMITE_Y_PIQUERA = 610

# Parámetros de estabilidad temporal utilizados para analizar y cerrar trayectorias.
VENTANA_ANALISIS = 5
FOTOGRAMAS_MINIMOS = 6
FOTOGRAMAS_DESAPARICION = 30

# Una trayectoria antigua se elimina de memoria después de superar este tiempo de ausencia. Este valor no interviene en la clasificación del evento.
FOTOGRAMAS_MAX_PERDIDA = 90

# EXTERIOR y ZONA_VUELO se consideran zonas situadas fuera de la piquera.
ZONAS_VUELO = {"EXTERIOR", "ZONA_VUELO"}

# Pesos utilizados por los trackers que necesitan información de apariencia.
RUTA_PESOS_REID = Path("osnet_x0_25_msmt17.pt")
if not RUTA_PESOS_REID.is_file():
    raise FileNotFoundError(f"No se encontraron los pesos de Re-ID: {RUTA_PESOS_REID}")

EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class TrayectoriaAbeja:
    """Guarda el historial asociado a un identificador devuelto por el tracker."""

    id_abeja: int
    historial: list = field(default_factory=list)
    ultimo_fotograma_visto: int = 0
    # Evita que una trayectoria ya cerrada vuelva a generar eventos.
    procesado: bool = False


def calcular_centroide(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int]:
    """Calcula el punto central de una caja delimitadora."""

    return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)


def clasificar_zona_geometrica(centroide: tuple[int, int]) -> str:
    """Clasifica un centroide como EXTERIOR, ZONA_VUELO o ZONA_PIQUERA."""

    x, y = centroide
    x1, y1, x2, y2 = CAJA_CONTEO

    # Todo punto que queda fuera de la caja de conteo se considera exterior.
    if x < x1 or x > x2 or y < y1 or y > y2:
        return "EXTERIOR"

    # Dentro de la caja, la coordenada vertical separa la zona de vuelo de la piquera.
    if y >= LIMITE_Y_PIQUERA:
        return "ZONA_PIQUERA"

    return "ZONA_VUELO"


def buscar_fotograma_transicion(historial: list, zonas_origen: set[str], zonas_destino: set[str]) -> int:
    """Devuelve el fotograma de la primera transición compatible entre las zonas de origen y destino."""

    zona_anterior = None

    for fotograma, _, zona_actual in historial:
        if zona_anterior in zonas_origen and zona_actual in zonas_destino:
            return fotograma

        zona_anterior = zona_actual

    # Si no aparece una transición explícita, se devuelve el último fotograma del historial.
    return historial[-1][0]


def analizar_eventos_trayectoria(historial: list) -> list[tuple[int, str]]:
    """Analiza el historial completo de una trayectoria y devuelve los eventos IN y OUT encontrados."""

    # Las trayectorias demasiado cortas no se utilizan para generar eventos.
    if len(historial) < FOTOGRAMAS_MINIMOS:
        return []

    ventana = max(3, VENTANA_ANALISIS)

    # La zona inicial y final se obtienen con la zona más repetida en los extremos del historial.
    zonas_inicio = [zona for _, _, zona in historial[:ventana]]
    zonas_fin = [zona for _, _, zona in historial[-ventana:]]

    if not zonas_inicio or not zonas_fin:
        return []

    zona_inicial = Counter(zonas_inicio).most_common(1)[0][0]
    zona_final = Counter(zonas_fin).most_common(1)[0][0]

    eventos = []

    # Si empieza y termina en la piquera, se comprueba si salió a una zona exterior y volvió a entrar.
    if zona_inicial == "ZONA_PIQUERA" and zona_final == "ZONA_PIQUERA":
        zonas_intermedias = [zona for _, _, zona in historial[ventana:-ventana] if zona in ZONAS_VUELO]

        # Deben existir al menos cinco observaciones intermedias fuera de la piquera, aunque no tienen que ser consecutivas.
        if len(zonas_intermedias) >= ventana:
            fotograma_salida = buscar_fotograma_transicion(historial, {"ZONA_PIQUERA"}, ZONAS_VUELO)
            eventos.append((fotograma_salida, "OUT"))

            fotograma_entrada = buscar_fotograma_transicion(historial, ZONAS_VUELO, {"ZONA_PIQUERA"})
            eventos.append((fotograma_entrada, "IN"))

    # Entrada estándar: la trayectoria empieza fuera de la piquera y termina dentro.
    elif zona_inicial in ZONAS_VUELO and zona_final == "ZONA_PIQUERA":
        fotograma_entrada = buscar_fotograma_transicion(historial, ZONAS_VUELO, {"ZONA_PIQUERA"})
        eventos.append((fotograma_entrada, "IN"))

    # Salida estándar: la trayectoria empieza en la piquera y termina fuera.
    elif zona_inicial == "ZONA_PIQUERA" and zona_final in ZONAS_VUELO:
        fotograma_salida = buscar_fotograma_transicion(historial, {"ZONA_PIQUERA"}, ZONAS_VUELO)
        eventos.append((fotograma_salida, "OUT"))

    return eventos


def evaluar_trayectoria_cerrada(trayectoria: TrayectoriaAbeja, fotograma_actual: int) -> list[tuple[int, str]]:
    """Procesa una trayectoria cuando lleva suficientes fotogramas sin aparecer."""

    # Una trayectoria ya procesada no debe volver a generar eventos.
    if trayectoria.procesado:
        return []

    # La trayectoria se considera cerrada cuando lleva 30 fotogramas sin aparecer.
    if (fotograma_actual - trayectoria.ultimo_fotograma_visto) >= FOTOGRAMAS_DESAPARICION:
        eventos = analizar_eventos_trayectoria(trayectoria.historial)
        trayectoria.procesado = True
        return eventos

    return []


def limpiar_memoria_trayectorias(memoria_trayectorias: dict[int, TrayectoriaAbeja], fotograma_actual: int, identificadores_activos: set[int]) -> None:
    """Elimina las trayectorias antiguas que llevan demasiado tiempo sin aparecer."""

    # Solo se eliminan los identificadores que no están activos y superan el límite de ausencia.
    identificadores_obsoletos = [
        id_abeja for id_abeja, trayectoria in memoria_trayectorias.items()
        if id_abeja not in identificadores_activos
        and (fotograma_actual - trayectoria.ultimo_fotograma_visto) > FOTOGRAMAS_MAX_PERDIDA
    ]

    for id_abeja in identificadores_obsoletos:
        del memoria_trayectorias[id_abeja]


def sincronizar_cuda() -> None:
    """Sincroniza la GPU antes o después de una medida de tiempo cuando se utiliza CUDA."""

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


def cargar_detecciones_video(ruta_csv: Path) -> dict[int, np.ndarray]:
    """Carga las detecciones de un vídeo y las agrupa por número de fotograma."""

    tabla_detecciones = pd.read_csv(ruta_csv)
    detecciones_por_fotograma = {}

    # Cada entrada del diccionario contiene todas las detecciones de un mismo fotograma.
    for numero_fotograma, grupo in tabla_detecciones.groupby("frame"):
        detecciones_por_fotograma[int(numero_fotograma)] = grupo[["x1", "y1", "x2", "y2", "conf", "cls"]].to_numpy(dtype=np.float32)
    
    return detecciones_por_fotograma


def ejecutar_benchmark(directorio_entrada: Path, directorio_detecciones: Path, directorio_salida: Path, algoritmos_seguimiento: list[str], fotogramas_warmup: int) -> None:
    """Ejecuta todos los trackers sobre las mismas detecciones y genera los eventos de cada vídeo."""

    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Benchmark de trackers con detecciones almacenadas")
    print(f"Vídeos: {directorio_entrada}")
    print(f"Detecciones: {directorio_detecciones}")
    print(f"Salida: {directorio_salida}")
    print(f"Trackers: {algoritmos_seguimiento}")
    print(f"Warm-up: {fotogramas_warmup} fotogramas")
    print(f"Dispositivo: {dispositivo}")

    directorio_salida.mkdir(parents=True, exist_ok=True)

    # Se recorren solo los archivos de vídeo con extensiones válidas.
    videos = sorted(ruta for ruta in directorio_entrada.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_VIDEO)

    if not videos:
        print(f"[ERROR] No se encontraron vídeos en: {directorio_entrada}")
        return

    resumen_por_video = []
    resumen_tiempos_trackers = []

    # Cada tracker procesa todos los vídeos usando exactamente los mismos CSV de detecciones.
    for nombre_algoritmo in algoritmos_seguimiento:
        print(f"\nEvaluando tracker: {nombre_algoritmo.upper()}")

        # Aquí se acumulan los tiempos de todos los vídeos del tracker actual.
        tiempos_globales_tracker: list[float] = []

        for ruta_video in videos:
            nombre_video = ruta_video.stem
            ruta_csv_detecciones = directorio_detecciones / f"detections_{nombre_video}.csv"

            # Cada vídeo necesita su archivo de detecciones generado previamente por 3A.
            if not ruta_csv_detecciones.exists():
                raise FileNotFoundError(f"No existe el archivo de detecciones requerido: {ruta_csv_detecciones}")

            print(f"Procesando {ruta_video.name} con {nombre_algoritmo}")

            # Las detecciones se cargan una sola vez antes de recorrer el vídeo.
            detecciones_por_fotograma = cargar_detecciones_video(ruta_csv_detecciones)

            # No se proporciona una configuración personalizada (None), por lo que cada algoritmo utiliza la configuración predeterminada de BoxMOT.
            # Los pesos OSNet se proporcionan para los trackers que utilizan apariencia.
            seguidor = create_tracker(nombre_algoritmo, None, str(RUTA_PESOS_REID), dispositivo, False)

            # El vídeo sigue siendo necesario porque algunos trackers utilizan también la imagen actual.
            captura_video = cv2.VideoCapture(str(ruta_video))

            if not captura_video.isOpened():
                raise FileNotFoundError( f"No se pudo abrir el vídeo: {ruta_video}")


            # Variables utilizadas únicamente durante el vídeo actual.
            registro_eventos: list[list] = []
            memoria_trayectorias: dict[int, TrayectoriaAbeja] = {}
            total_entradas = 0
            total_salidas = 0
            indice_fotograma = 0
            tiempos_seguimiento_ms: list[float] = []

            barra_progreso = tqdm(desc=f"Tracking {nombre_video} [{nombre_algoritmo}]", unit="frame")

            while captura_video.isOpened():
                lectura_correcta, fotograma = captura_video.read()

                if not lectura_correcta:
                    break

                indice_fotograma += 1
                medir_tiempo = indice_fotograma > fotogramas_warmup

                # Se recuperan las detecciones del fotograma actual.
                # Si no hay ninguna, se utiliza una matriz vacía.
                cajas_detectadas = detecciones_por_fotograma.get(indice_fotograma, np.empty((0, 6)))

                if medir_tiempo:
                    sincronizar_cuda()
                    tiempo_inicio = time.perf_counter()

                # update se ejecuta siempre para que el tracker avance también en fotogramas sin detecciones.
                seguimientos_actualizados = seguidor.update(cajas_detectadas, fotograma)

                # El cronómetro cubre únicamente tracker.update(...).
                # Quedan fuera la lectura del vídeo, la carga de detecciones, la gestión de historiales, la lógica de eventos y la escritura de CSV.
                if medir_tiempo:
                    sincronizar_cuda()
                    tiempo_fin = time.perf_counter()
                    tiempos_seguimiento_ms.append((tiempo_fin - tiempo_inicio) * 1000)

                # Guarda los identificadores que han aparecido en el fotograma actual.
                identificadores_activos: set[int] = set()

                # Se actualiza el historial de cada trayectoria devuelta por el tracker.
                if seguimientos_actualizados is not None and len(seguimientos_actualizados) > 0:
                    for seguimiento in seguimientos_actualizados:
                        
                        # Las primeras posiciones devueltas por BoxMOT contienen la caja xyxy y el identificador de la trayectoria.
                        x1, y1, x2, y2, id_abeja = map(float, seguimiento[:5])
                        id_abeja = int(id_abeja)

                        centroide = calcular_centroide(x1, y1, x2, y2)
                        zona_actual = clasificar_zona_geometrica(centroide)

                        # Si el identificador es nuevo o su trayectoria anterior ya terminó, se crea un historial nuevo.
                        if id_abeja not in memoria_trayectorias or memoria_trayectorias[id_abeja].procesado:
                            memoria_trayectorias[id_abeja] = TrayectoriaAbeja(id_abeja=id_abeja)

                        trayectoria = memoria_trayectorias[id_abeja]

                        # Se guarda el fotograma, el centroide y la zona observada.
                        trayectoria.historial.append((indice_fotograma, centroide, zona_actual))
                        trayectoria.ultimo_fotograma_visto = indice_fotograma
                        identificadores_activos.add(id_abeja)

                # Las trayectorias que no aparecen en este fotograma se revisan para comprobar si deben cerrarse.
                for id_abeja, trayectoria in list(memoria_trayectorias.items()):
                    if id_abeja in identificadores_activos:
                        continue

                    nuevos_eventos = evaluar_trayectoria_cerrada(trayectoria, indice_fotograma)

                    # Los eventos encontrados se añaden al registro del vídeo y actualizan los contadores.
                    for fotograma_cruce, tipo_evento in nuevos_eventos:
                        if tipo_evento == "IN":
                            total_entradas += 1
                        elif tipo_evento == "OUT":
                            total_salidas += 1

                        registro_eventos.append([fotograma_cruce, tipo_evento])

                # Las trayectorias antiguas se eliminan cuando superan el tiempo máximo de ausencia.
                limpiar_memoria_trayectorias(memoria_trayectorias, indice_fotograma, identificadores_activos)

                barra_progreso.update(1)

            barra_progreso.close()
            captura_video.release()

            # Al terminar el vídeo también se procesan las trayectorias que todavía estaban abiertas.
            for id_abeja, trayectoria in list(memoria_trayectorias.items()):
                if not trayectoria.procesado:
                    trayectoria.procesado = True
                    eventos_cierre = analizar_eventos_trayectoria(trayectoria.historial)

                    for fotograma_cruce, tipo_evento in eventos_cierre:
                        if tipo_evento == "IN":
                            total_entradas += 1
                        elif tipo_evento == "OUT":
                            total_salidas += 1

                        registro_eventos.append([fotograma_cruce, tipo_evento])

            # Las medidas de tiempo se calculan solo con los fotogramas posteriores al warm-up.
            tiempo_medio_ms = calcular_media(tiempos_seguimiento_ms)
            tiempo_mediano_ms = calcular_mediana(tiempos_seguimiento_ms)
            desviacion_tiempo_ms = calcular_desviacion(tiempos_seguimiento_ms)
            fps_tracker = 1000.0 / tiempo_medio_ms if tiempo_medio_ms > 0 else 0.0

            # Los tiempos de este vídeo se añaden para calcular después el resultado global del tracker.
            tiempos_globales_tracker.extend(tiempos_seguimiento_ms)

            ruta_csv_eventos = directorio_salida / f"results_{nombre_algoritmo}_{nombre_video}.csv"

            # Los eventos se ordenan por fotograma antes de guardar el CSV que utilizará el evaluador temporal.
            with ruta_csv_eventos.open("w", newline="", encoding="utf-8") as archivo_csv:
                escritor = csv.writer(archivo_csv)
                escritor.writerow(["frame", "event"])
                escritor.writerows(sorted(registro_eventos, key=lambda evento: evento[0]))

            # Se guarda una fila de resumen para la combinación actual de tracker y vídeo.
            resumen_por_video.append([
                nombre_algoritmo,
                nombre_video,
                total_entradas,
                total_salidas,
                total_entradas + total_salidas,
                len(registro_eventos),
                tiempo_medio_ms,
                tiempo_mediano_ms,
                desviacion_tiempo_ms,
                fps_tracker,
                str(ruta_csv_eventos),
            ])

            print(f"IN: {total_entradas} | OUT: {total_salidas} | media={tiempo_medio_ms:.1f} ms | mediana={tiempo_mediano_ms:.1f} ms | desv={desviacion_tiempo_ms:.1f} ms | {fps_tracker:.1f} FPS")
            print(f"Eventos guardados en: {ruta_csv_eventos}")

        # Una vez terminados todos los vídeos se calculan las medidas globales del tracker actual.
        tiempo_global_medio_ms = calcular_media(tiempos_globales_tracker)
        tiempo_global_mediano_ms = calcular_mediana(tiempos_globales_tracker)
        desviacion_global_ms = calcular_desviacion(tiempos_globales_tracker)
        fps_global = 1000.0 / tiempo_global_medio_ms if tiempo_global_medio_ms > 0 else 0.0

        resumen_tiempos_trackers.append([
            nombre_algoritmo,
            len(tiempos_globales_tracker),
            tiempo_global_medio_ms,
            tiempo_global_mediano_ms,
            desviacion_global_ms,
            fps_global,
        ])

        print(f"Global {nombre_algoritmo.upper()}: fotogramas medidos={len(tiempos_globales_tracker)} | media={tiempo_global_medio_ms:.3f} ms | mediana={tiempo_global_mediano_ms:.3f} ms | desv={desviacion_global_ms:.3f} ms | {fps_global:.2f} FPS")

    # Resumen con los conteos y tiempos de cada combinación de tracker y vídeo.
    ruta_resumen = directorio_salida / "summary_by_tracker.csv"

    with ruta_resumen.open("w", newline="", encoding="utf-8") as archivo_resumen:
        escritor = csv.writer(archivo_resumen)
        escritor.writerow(["tracker", "video", "IN_pred", "OUT_pred", "total_pred", "num_events", "avg_trk_ms", "median_trk_ms", "std_trk_ms", "tracker_fps", "events_csv"])
        escritor.writerows(resumen_por_video)

    # Resumen temporal calculado con todos los fotogramas medidos de cada tracker.
    ruta_resumen_tiempos = directorio_salida / "summary_timing_global_by_tracker.csv"

    with ruta_resumen_tiempos.open("w", newline="", encoding="utf-8") as archivo_resumen_tiempos:
        escritor = csv.writer(archivo_resumen_tiempos)
        escritor.writerow(["tracker", "num_timed_frames", "avg_trk_ms", "median_trk_ms", "std_trk_ms", "tracker_fps"])
        escritor.writerows(resumen_tiempos_trackers)

    print(f"Resumen por vídeo guardado en: {ruta_resumen}")
    print(f"Resumen temporal global guardado en: {ruta_resumen_tiempos}")
    print("Benchmark finalizado.")


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Compara varios trackers utilizando detecciones almacenadas previamente.")

    analizador.add_argument("--input", dest="directorio_entrada", type=Path, default=Path("data/videos_evaluacion_temporal"), help="Carpeta que contiene los vídeos de evaluación temporal.")
    analizador.add_argument("--detections", dest="directorio_detecciones", type=Path, default=Path("results/detections"), help="Carpeta con los CSV de detecciones generados por 3A.")
    analizador.add_argument("--output", dest="directorio_salida", type=Path, default=Path("results/benchmark"), help="Carpeta donde se guardarán los eventos y los resúmenes.")
    analizador.add_argument("--trackers", dest="algoritmos_seguimiento", type=str, nargs="+", default=["bytetrack", "botsort", "ocsort", "boosttrack", "strongsort"], help="Lista de trackers que se van a evaluar.")
    analizador.add_argument("--warmup_frames", dest="fotogramas_warmup", type=int, default=30, help="Fotogramas iniciales excluidos del cálculo de tiempos.")

    argumentos = analizador.parse_args()

    # Se comprueba que las entradas necesarias existen antes de comenzar.
    if not argumentos.directorio_entrada.is_dir():
        analizador.error(f"No existe la carpeta de vídeos: {argumentos.directorio_entrada}")

    if not argumentos.directorio_detecciones.is_dir():
        analizador.error(f"No existe la carpeta de detecciones: {argumentos.directorio_detecciones}")

    if argumentos.fotogramas_warmup < 0:
        analizador.error("--warmup_frames debe ser mayor o igual que 0.")

    return argumentos


def main() -> None:
    """Lee los argumentos y ejecuta el benchmark de trackers."""

    argumentos = obtener_argumentos()

    ejecutar_benchmark(
        directorio_entrada=argumentos.directorio_entrada,
        directorio_detecciones=argumentos.directorio_detecciones,
        directorio_salida=argumentos.directorio_salida,
        algoritmos_seguimiento=argumentos.algoritmos_seguimiento,
        fotogramas_warmup=argumentos.fotogramas_warmup,
    )


if __name__ == "__main__":
    main()