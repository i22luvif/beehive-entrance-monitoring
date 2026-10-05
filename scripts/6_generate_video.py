"""
Genera un vídeo demostrativo del funcionamiento de Bee-TFG.

El script ejecuta de forma independiente el detector YOLO definitivo y
ByteTrack sobre un vídeo. Las detecciones no se leen de los CSV utilizados
durante el benchmark, ya que esta ejecución tiene únicamente finalidad visual.

Sobre cada fotograma se muestran:

    - cajas delimitadoras;
    - centroides;
    - estelas recientes de las trayectorias;
    - zona de vuelo y zona de piquera;
    - número de fotograma;
    - contador acumulado de eventos IN;
    - contador acumulado de eventos OUT.

La lógica espacial y los principales parámetros temporales coinciden con los
utilizados en el benchmark, pero este script no genera archivos de eventos ni
interviene en las métricas publicadas.

El vídeo se guarda por defecto en:

    results/videos/demo.mp4
"""

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from boxmot import create_tracker
from ultralytics import YOLO


# Geometría utilizada para separar la zona de vuelo y la piquera.
CAJA_CONTEO = (40, 560, 1210, 635)
LIMITE_Y_PIQUERA = 610
CAJA_PIQUERA = (CAJA_CONTEO[0], LIMITE_Y_PIQUERA, CAJA_CONTEO[2], CAJA_CONTEO[3])

# Parámetros utilizados para analizar y cerrar las trayectorias.
VENTANA_ANALISIS = 5
FOTOGRAMAS_MINIMOS = 6
FOTOGRAMAS_DESAPARICION = 30
# Los estados antiguos se eliminan de memoria después de superar este tiempo de ausencia. Este valor no interviene en la clasificación.
FOTOGRAMAS_MAX_PERDIDA = 90

ZONAS_VUELO = {"EXTERIOR", "ZONA_VUELO"}

# Parámetros utilizados durante la inferencia.
UMBRAL_CONFIANZA = 0.25
RESOLUCION = 960
TAMANO_MINIMO_CAJA = 10

# La interfaz de BoxMOT recibe una ruta de pesos Re-ID, aunque ByteTrack no utiliza características de apariencia.
RUTA_PESOS_REID = Path("osnet_x0_25_msmt17.pt")

# Configuración visual del vídeo generado.
COLOR_TEXTO = (255, 255, 255)
COLOR_ESTELA = (200, 100, 200)
TRANSPARENCIA_ZONA_VUELO = 0.10
TRANSPARENCIA_ZONA_PIQUERA = 0.10
TRANSPARENCIA_PANEL = 0.60
LONGITUD_ESTELA = 30


@dataclass
class TrayectoriaVisual:
    """Guarda el historial y los puntos necesarios para dibujar una trayectoria."""

    id_abeja: int
    historial: list = field(default_factory=list)
    puntos_dibujo: deque = field(default_factory=lambda: deque(maxlen=LONGITUD_ESTELA))
    ultimo_fotograma: int = 0
    procesada: bool = False


def calcular_centroide(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int]:
    """Calcula el punto central de una caja delimitadora."""

    return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)


def clasificar_zona(centroide: tuple[int, int]) -> str:
    """Clasifica un centroide como EXTERIOR, ZONA_VUELO o ZONA_PIQUERA."""

    x, y = centroide
    x1, y1, x2, y2 = CAJA_CONTEO

    if x < x1 or x > x2 or y < y1 or y > y2:
        return "EXTERIOR"

    if y >= LIMITE_Y_PIQUERA:
        return "ZONA_PIQUERA"

    return "ZONA_VUELO"


def analizar_evento(historial: list) -> str | None:
    """Analiza una trayectoria cerrada y devuelve el tipo de evento detectado."""

    # Las trayectorias muy cortas no tienen suficientes observaciones para generar un evento.
    if len(historial) < FOTOGRAMAS_MINIMOS:
        return None

    ventana = max(3, VENTANA_ANALISIS)
    zonas_inicio = [zona for _, _, zona in historial[:ventana]]
    zonas_fin = [zona for _, _, zona in historial[-ventana:]]

    if not zonas_inicio or not zonas_fin:
        return None

    # Se utiliza la zona más repetida al principio y al final para reducir cambios puntuales de posición.
    zona_inicial = Counter(zonas_inicio).most_common(1)[0][0]
    zona_final = Counter(zonas_fin).most_common(1)[0][0]

    # Si empieza y termina en la piquera, se comprueba si salió a la zona exterior y regresó.
    if zona_inicial == "ZONA_PIQUERA" and zona_final == "ZONA_PIQUERA":
        zonas_intermedias = [zona for _, _, zona in historial[ventana:-ventana] if zona in ZONAS_VUELO]
        # Deben existir al menos cinco observaciones intermedias fuera de la piquera; no es necesario que sean consecutivas.
        if len(zonas_intermedias) >= ventana:
            return "OUT_IN"

    # Entrada estándar: exterior o vuelo hacia la piquera.
    elif zona_inicial in ZONAS_VUELO and zona_final == "ZONA_PIQUERA":
        return "IN"

    # Salida estándar: piquera hacia exterior o vuelo.
    elif zona_inicial == "ZONA_PIQUERA" and zona_final in ZONAS_VUELO:
        return "OUT"

    return None


def aplicar_rectangulo_transparente(fotograma: np.ndarray, punto_1: tuple[int, int], punto_2: tuple[int, int], color: tuple[int, int, int], transparencia: float) -> None:
    """Dibuja un rectángulo semitransparente sobre el fotograma."""

    capa = fotograma.copy()
    cv2.rectangle(capa, punto_1, punto_2, color, -1)
    cv2.addWeighted(capa, transparencia, fotograma, 1 - transparencia, 0, fotograma)


def renderizar_panel(fotograma: np.ndarray, indice_fotograma: int, entradas: int, salidas: int) -> np.ndarray:
    """Dibuja las zonas de conteo y el panel con los contadores acumulados."""

    x1, y1, x2, y2 = CAJA_CONTEO
    px1, py1, px2, py2 = CAJA_PIQUERA

    # Las dos zonas se dibujan con transparencia para no ocultar las abejas.
    aplicar_rectangulo_transparente(fotograma, (x1, y1), (x2, py1), (255, 0, 0), TRANSPARENCIA_ZONA_VUELO)
    aplicar_rectangulo_transparente(fotograma, (px1, py1), (px2, py2), (0, 255, 0), TRANSPARENCIA_ZONA_PIQUERA)

    # Los bordes muestran los límites utilizados por la lógica de conteo.
    cv2.rectangle(fotograma, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(fotograma, (x1, py1), (x2, py1), (255, 255, 255), 1, cv2.LINE_AA)

    # El panel superior muestra el número de fotograma y los contadores acumulados.
    capa_panel = fotograma.copy()
    ancho_panel = min(fotograma.shape[1], 1280)
    cv2.rectangle(capa_panel, (0, 0), (ancho_panel, 60), (0, 0, 0), -1)
    cv2.addWeighted(capa_panel, TRANSPARENCIA_PANEL, fotograma, 1 - TRANSPARENCIA_PANEL, 0, fotograma)

    texto_panel = f"TFG DEMO | Frame: {indice_fotograma:04d} | ENTRADAS (IN): {entradas} | SALIDAS (OUT): {salidas}"
    cv2.putText(fotograma, texto_panel, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXTO, 2, cv2.LINE_AA)

    return fotograma


def generar_video_demostrativo(ruta_modelo: Path, ruta_video: Path, ruta_salida: Path) -> None:
    """Genera un vídeo con detecciones, trayectorias, zonas y contadores IN/OUT."""

    print("Generando vídeo demostrativo...")

    # El detector se ejecuta directamente sobre cada fotograma del vídeo.
    modelo = YOLO(str(ruta_modelo))

    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ByteTrack se crea mediante BoxMOT sin una configuración personalizada, por lo que utiliza los valores predeterminados de la versión instalada.
    tracker = create_tracker("bytetrack", None, str(RUTA_PESOS_REID), dispositivo, False)

    # Se abre el vídeo y se leen la resolución y los FPS originales.
    captura_video = cv2.VideoCapture(str(ruta_video))

    if not captura_video.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo de entrada: {ruta_video}")

    ancho_fotograma = int(captura_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_fotograma = int(captura_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = captura_video.get(cv2.CAP_PROP_FPS)

    # Si OpenCV no puede recuperar los FPS, se utiliza 25 como valor de respaldo.
    if fps_video <= 0:
        fps_video = 25

    # Si ya existe un vídeo con la misma ruta de salida la nueva ejecución puede sustituirlo.
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    # El vídeo de salida conserva la resolución y la tasa de fotogramas del original.
    escritor_video = cv2.VideoWriter(str(ruta_salida), cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (ancho_fotograma, alto_fotograma))

    if not escritor_video.isOpened():
        captura_video.release()
        raise RuntimeError(f"No se pudo crear el vídeo de salida: {ruta_salida}")

    trayectorias: dict[int, TrayectoriaVisual] = {}
    total_entradas = 0
    total_salidas = 0
    indice_fotograma = 0

    try:
        while captura_video.isOpened():
            lectura_correcta, fotograma = captura_video.read()

            if not lectura_correcta:
                break

            indice_fotograma += 1

            # Se ejecuta YOLO con la confianza y resolución definidas para la inferencia.
            resultados = modelo(fotograma, verbose=False, conf=UMBRAL_CONFIANZA, imgsz=RESOLUCION)[0]
            cajas_filtradas = np.empty((0, 6))

            if len(resultados.boxes) > 0:
                cajas_brutas = np.hstack([
                    resultados.boxes.xyxy.cpu().numpy(),
                    resultados.boxes.conf.cpu().numpy().reshape(-1, 1),
                    resultados.boxes.cls.cpu().numpy().reshape(-1, 1),
                ])

                # Se descartan las cajas con menos de 10 píxeles de ancho o alto.
                mascara_tamano = (
                    (cajas_brutas[:, 2] - cajas_brutas[:, 0] >= TAMANO_MINIMO_CAJA)
                    & (cajas_brutas[:, 3] - cajas_brutas[:, 1] >= TAMANO_MINIMO_CAJA)
                )

                cajas_filtradas = cajas_brutas[mascara_tamano]

            # El tracker se actualiza incluso cuando no hay detecciones en el fotograma.
            seguimientos = tracker.update(cajas_filtradas, fotograma)
            identificadores_activos: set[int] = set()

            if seguimientos is not None and len(seguimientos) > 0:
                for seguimiento in seguimientos:
                    x1, y1, x2, y2, id_abeja = map(float, seguimiento[:5])
                    id_abeja = int(id_abeja)

                    centroide = calcular_centroide(x1, y1, x2, y2)
                    zona = clasificar_zona(centroide)

                    # Si el identificador es nuevo o su trayectoria anterior ya fue procesada, se crea una nueva.
                    if id_abeja not in trayectorias or trayectorias[id_abeja].procesada:
                        trayectorias[id_abeja] = TrayectoriaVisual(id_abeja=id_abeja)

                    trayectoria = trayectorias[id_abeja]
                    trayectoria.historial.append((indice_fotograma, centroide, zona))
                    trayectoria.puntos_dibujo.append(centroide)
                    trayectoria.ultimo_fotograma = indice_fotograma
                    identificadores_activos.add(id_abeja)

                    # Se dibujan la caja, la estela reciente y el centroide de la trayectoria.
                    cv2.rectangle(fotograma, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1)

                    for indice_punto in range(1, len(trayectoria.puntos_dibujo)):
                        grosor_estela = int(np.sqrt(LONGITUD_ESTELA / float(LONGITUD_ESTELA - indice_punto + 1)) * 1.5)
                        cv2.line(fotograma, trayectoria.puntos_dibujo[indice_punto - 1], trayectoria.puntos_dibujo[indice_punto], COLOR_ESTELA, grosor_estela)

                    cv2.circle(fotograma, centroide, 3, (0, 255, 255), -1)

            # Se revisan las trayectorias que no han aparecido en el fotograma actual.
            for id_abeja, trayectoria in list(trayectorias.items()):
                fotogramas_ausente = indice_fotograma - trayectoria.ultimo_fotograma

                # Tras 30 fotogramas de ausencia se cierra la trayectoria y se actualizan los contadores. No se busca el fotograma exacto de la transición,
                # ya que este script no genera eventos para la evaluación temporal.
                if id_abeja not in identificadores_activos and not trayectoria.procesada and fotogramas_ausente >= FOTOGRAMAS_DESAPARICION:
                    evento = analizar_evento(trayectoria.historial)
                    trayectoria.procesada = True

                    if evento == "IN":
                        total_entradas += 1
                    elif evento == "OUT":
                        total_salidas += 1
                    # OUT_IN representa una salida seguida de una entrada, por lo que se incrementan ambos contadores.
                    elif evento == "OUT_IN":
                        total_salidas += 1
                        total_entradas += 1

                # Los estados antiguos se eliminan cuando llevan más de 90 fotogramas sin aparecer.
                if id_abeja not in identificadores_activos and fotogramas_ausente > FOTOGRAMAS_MAX_PERDIDA:
                    del trayectorias[id_abeja]

            # Las trayectorias que siguen abiertas al terminar el vídeo no se fuerzan a cerrar en este módulo demostrativo.
            fotograma_final = renderizar_panel(fotograma, indice_fotograma, total_entradas, total_salidas)
            escritor_video.write(fotograma_final)

    finally:
        captura_video.release()
        escritor_video.release()

    print(f"Vídeo demostrativo guardado en: {ruta_salida}")


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Genera un vídeo demostrativo con YOLO y ByteTrack.")

    analizador.add_argument("--model", dest="ruta_modelo", type=Path, default=Path("model/best_yolo.pt"), help="Ruta al detector YOLO utilizado para generar el vídeo.")
    analizador.add_argument("--video", dest="ruta_video", type=Path, required=True, help="Ruta al vídeo que se quiere procesar.")
    analizador.add_argument("--output", dest="ruta_salida", type=Path, default=Path("results/videos/demo1.mp4"), help="Ruta donde se guardará el vídeo generado.")

    argumentos = analizador.parse_args()

    # Se comprueba que los archivos de entrada existen antes de empezar.
    if not argumentos.ruta_modelo.is_file():
        analizador.error(f"No existe el modelo: {argumentos.ruta_modelo}")

    if not argumentos.ruta_video.is_file():
        analizador.error(f"No existe el vídeo: {argumentos.ruta_video}")

    return argumentos


def main() -> None:
    """Lee los argumentos y genera el vídeo demostrativo."""

    argumentos = obtener_argumentos()
    generar_video_demostrativo(ruta_modelo=argumentos.ruta_modelo, ruta_video=argumentos.ruta_video, ruta_salida=argumentos.ruta_salida)


if __name__ == "__main__":
    main()