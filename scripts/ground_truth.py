"""
Herramienta interactiva para crear la referencia temporal manual del sistema.

El script reproduce un vídeo y permite registrar eventos IN y OUT indicando
el fotograma en el que se observan. Las anotaciones se almacenan con el formato:

    frame,event

El detector YOLO definitivo se utiliza como apoyo visual para localizar
las abejas, pero no genera eventos automáticamente. Las zonas mostradas en
pantalla corresponden a la geometría utilizada posteriormente por la lógica
de conteo.

Controles principales:

    espacio  Reproducir o pausar
    a / d    Retroceder o avanzar 1 fotograma
    b / n    Retroceder o avanzar 25 fotogramas
    i        Registrar evento IN
    o        Registrar evento OUT
    u        Deshacer el último evento
    s        Guardar
    q        Guardar y salir
    + / -    Aumentar o reducir la velocidad de reproducción
"""


import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# Coordenadas de las zonas que se muestran durante la anotación.
CAJA_ANALISIS = (40, 560, 1210, 635)
LINEA_FRONTERA_PIQUERA = 610
CAJA_PIQUERA = (CAJA_ANALISIS[0], LINEA_FRONTERA_PIQUERA, CAJA_ANALISIS[2], CAJA_ANALISIS[3])

# Tamaño mínimo de las detecciones que se muestran como ayuda visual.
TAMANO_MINIMO_ABEJA = 10
VERSION_INTERFAZ = "Bee-TFG Ground Truth"


def asegurar_directorio(ruta_archivo: Path) -> None:
    """Crea la carpeta de destino si todavía no existe."""
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)


def guardar_eventos(ruta_salida: Path, eventos: list[dict]) -> None:
    """Guarda los eventos anotados en formato frame,event."""
    asegurar_directorio(ruta_salida)

    with ruta_salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["frame", "event"])
        escritor.writeheader()

        # Los eventos se ordenan por fotograma antes de escribir el archivo.
        for evento in sorted(eventos, key=lambda elemento: int(elemento["fotograma"])):
            escritor.writerow({"frame": evento["fotograma"], "event": evento["evento"]})


def inicializar_modelo_yolo(ruta_modelo: Path | None):
    """Carga el modelo YOLO utilizado como apoyo visual durante la anotación."""
    if ruta_modelo is None:
        return None

    if YOLO is None:
        raise RuntimeError("No se pudo importar Ultralytics.")

    print(f"Cargando modelo de asistencia visual: {ruta_modelo}")
    return YOLO(str(ruta_modelo))


def obtener_abejas_detectadas(modelo, fotograma: np.ndarray, confianza_minima: float) -> np.ndarray:
    """Ejecuta YOLO y devuelve las cajas válidas que se mostrarán en pantalla."""
    if modelo is None:
        return np.empty((0, 5), dtype=float)

    resultados = modelo(fotograma, verbose=False, conf=confianza_minima)[0]

    if len(resultados.boxes) == 0:
        return np.empty((0, 5), dtype=float)

    # Cada fila contiene las coordenadas de la caja y la confianza de la detección.
    coordenadas = resultados.boxes.xyxy.cpu().numpy()
    confianzas = resultados.boxes.conf.cpu().numpy().reshape(-1, 1)
    detecciones = np.hstack([coordenadas, confianzas])

    # Se descartan las cajas con menos de 10 píxeles de ancho o alto.
    anchos = detecciones[:, 2] - detecciones[:, 0]
    altos = detecciones[:, 3] - detecciones[:, 1]
    mascara_valida = (anchos >= TAMANO_MINIMO_ABEJA) & (altos >= TAMANO_MINIMO_ABEJA)

    return detecciones[mascara_valida]


def calcular_centroide(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int]:
    """Calcula el punto central de una caja delimitadora."""
    centro_x = int(round((float(x1) + float(x2)) / 2.0))
    centro_y = int(round((float(y1) + float(y2)) / 2.0))
    return centro_x, centro_y


def clasificar_zona(punto: tuple[int, int]) -> str:
    """Indica en qué zona visual se encuentra un punto."""
    x, y = punto
    x_min, y_min, x_max, y_max = CAJA_ANALISIS

    if x < x_min or x > x_max or y < y_min or y > y_max:
        return "EXTERIOR"

    if y >= LINEA_FRONTERA_PIQUERA:
        return "ZONA_PIQUERA"

    return "ZONA_VUELO"


def dibujar_rectangulo_transparente(imagen: np.ndarray, punto_1: tuple[int, int], punto_2: tuple[int, int], color: tuple[int, int, int], transparencia: float = 0.12) -> None:
    """Dibuja un rectángulo semitransparente sobre una imagen."""
    capa = imagen.copy()
    cv2.rectangle(capa, punto_1, punto_2, color, -1)
    cv2.addWeighted(capa, transparencia, imagen, 1 - transparencia, 0, imagen)


def dibujar_interfaz(fotograma: np.ndarray, abejas_detectadas: np.ndarray, numero_fotograma: int, total_fotogramas: int, historial_eventos: list[dict], esta_en_pausa: bool, latencia_reproduccion: int, ultimo_click: tuple[int, int] | None) -> np.ndarray:
    """Dibuja las zonas, detecciones y controles sobre el fotograma actual."""
    lienzo = fotograma.copy()
    _, ancho_imagen = lienzo.shape[:2]

    x_min, y_min, x_max, y_max = CAJA_ANALISIS
    piquera_x_min, piquera_y_min, piquera_x_max, piquera_y_max = CAJA_PIQUERA

    # Se muestran las zonas de vuelo y piquera con una transparencia ligera.
    dibujar_rectangulo_transparente(lienzo, (x_min, y_min), (x_max, piquera_y_min - 1), (255, 0, 0), transparencia=0.06)
    dibujar_rectangulo_transparente(lienzo, (piquera_x_min, piquera_y_min), (piquera_x_max, piquera_y_max), (0, 255, 0), transparencia=0.10)
    cv2.rectangle(lienzo, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    cv2.rectangle(lienzo, (piquera_x_min, piquera_y_min), (piquera_x_max, piquera_y_max), (0, 255, 0), 1)

    # Las detecciones de YOLO se dibujan únicamente como ayuda para localizar las abejas.
    for abeja in abejas_detectadas:
        x1, y1, x2, y2, confianza = abeja
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        centroide = calcular_centroide(x1, y1, x2, y2)
        zona_actual = clasificar_zona(centroide)

        cv2.rectangle(lienzo, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.circle(lienzo, centroide, 3, (0, 255, 0), -1)
        etiqueta = f"YOLO  {confianza:.2f} [{zona_actual[:3]}]"
        cv2.putText(lienzo, etiqueta, (x1, max(55, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    # El último clic del ratón se marca con una cruz para tener una referencia visual.
    if ultimo_click:
        cv2.drawMarker(lienzo, ultimo_click, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    # El panel superior muestra el estado de reproducción, los contadores y los controles disponibles.
    dibujar_rectangulo_transparente(lienzo, (0, 0), (min(ancho_imagen, 1280), 85), (0, 0, 0), transparencia=0.70)

    total_in = sum(1 for evento in historial_eventos if evento["evento"] == "IN")
    total_out = sum(1 for evento in historial_eventos if evento["evento"] == "OUT")
    estado_reproduccion = "PAUSA" if esta_en_pausa else "PLAY"

    textos_panel = [
        f"{VERSION_INTERFAZ} | Frame: {numero_fotograma}/{total_fotogramas} | {estado_reproduccion} | Delay: {latencia_reproduccion}ms | IN: {total_in} | OUT: {total_out}",
        "[i] IN (Entra a piquera) | [o] OUT (Sale de piquera)",
        "Controles: [espacio] Play/Pausa | [a/d] +/- 1 Frame | [b/n] +/- 25 Frames | [+/-] Velocidad | [u] Deshacer | [s] Guardar | [q] Salir",
    ]

    for indice, texto in enumerate(textos_panel):
        posicion_y = 22 + indice * 18
        color_texto = (200, 255, 200) if indice == 0 else (220, 220, 220)
        cv2.putText(lienzo, texto, (20, posicion_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_texto, 1, cv2.LINE_AA)

    return lienzo


def iniciar_anotador(ruta_video: Path, ruta_salida: Path, ruta_modelo: Path | None, confianza_modelo: float) -> None:
    """Abre el vídeo y permite registrar manualmente eventos IN y OUT."""
    modelo_asistencia = inicializar_modelo_yolo(ruta_modelo)

    captura_video = cv2.VideoCapture(str(ruta_video))

    if not captura_video.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {ruta_video}")

    total_fotogramas = int(captura_video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_fotogramas <= 0:
        raise RuntimeError(f"El vídeo no contiene fotogramas válidos: {ruta_video}")

    # Estado que se mantiene mientras se revisa el vídeo.
    historial_eventos: list[dict] = []
    memoria_detecciones: dict[int, np.ndarray] = {}
    indice_actual = 0
    esta_en_pausa = True
    latencia_reproduccion = 40
    ultimo_click: tuple[int, int] | None = None

    nombre_ventana = "Bee-TFG - Anotación Ground Truth"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)

    def evento_raton(evento, x, y, _flags, _param) -> None:
        nonlocal ultimo_click

        if evento == cv2.EVENT_LBUTTONDOWN:
            ultimo_click = (int(x), int(y))

    cv2.setMouseCallback(nombre_ventana, evento_raton)

    def registrar_evento(tipo_evento: str, fotograma: int) -> None:
        nonlocal ultimo_click

        # Cada evento se guarda con el número de fotograma que se está mostrando en pantalla.
        historial_eventos.append({"fotograma": int(fotograma), "evento": tipo_evento})
        guardar_eventos(ruta_salida, historial_eventos)
        print(f"[{fotograma}] Registrado: {tipo_evento}")
        ultimo_click = None

    print("Iniciando entorno de anotación")

    try:
        while True:
            # El índice se mantiene siempre dentro de los límites del vídeo.
            indice_actual = max(0, min(indice_actual, total_fotogramas - 1))
            # OpenCV utiliza internamente índices desde 0, pero los eventos se numeran desde 1 para coincidir con los CSV de detecciones y benchmark.
            numero_fotograma = indice_actual + 1

            # Se coloca la captura en el fotograma solicitado para poder avanzar y retroceder libremente.
            captura_video.set(cv2.CAP_PROP_POS_FRAMES, indice_actual)
            lectura_correcta, fotograma = captura_video.read()

            if not lectura_correcta:
                break

            # Las detecciones se guardan en memoria para no repetir la inferencia al volver a un fotograma ya visto.
            if numero_fotograma not in memoria_detecciones:
                memoria_detecciones[numero_fotograma] = obtener_abejas_detectadas(modelo_asistencia, fotograma, confianza_modelo)

            abejas_en_pantalla = memoria_detecciones[numero_fotograma]

            pantalla = dibujar_interfaz(
                fotograma,
                abejas_en_pantalla,
                numero_fotograma,
                total_fotogramas,
                historial_eventos,
                esta_en_pausa,
                latencia_reproduccion,
                ultimo_click,
            )

            cv2.imshow(nombre_ventana, pantalla)

            # En pausa se espera hasta pulsar una tecla; durante la reproducción se usa la latencia configurada.
            tecla = cv2.waitKey(0 if esta_en_pausa else latencia_reproduccion) & 0xFF

            if tecla == ord("q"):
                guardar_eventos(ruta_salida, historial_eventos)
                print("Cerrando y guardando...")
                break

            elif tecla == ord(" "):
                esta_en_pausa = not esta_en_pausa

            elif tecla == ord("d"):
                esta_en_pausa = True
                indice_actual += 1

            elif tecla == ord("a"):
                esta_en_pausa = True
                indice_actual -= 1

            elif tecla == ord("n"):
                esta_en_pausa = True
                indice_actual += 25

            elif tecla == ord("b"):
                esta_en_pausa = True
                indice_actual -= 25

            elif tecla == ord("+"):
                latencia_reproduccion = max(5, latencia_reproduccion - 5)

            elif tecla == ord("-"):
                latencia_reproduccion = min(500, latencia_reproduccion + 5)

            elif tecla == ord("i"):
                registrar_evento("IN", numero_fotograma)

            elif tecla == ord("o"):
                registrar_evento("OUT", numero_fotograma)

            elif tecla == ord("s"):
                guardar_eventos(ruta_salida, historial_eventos)

            elif tecla == ord("u") and historial_eventos:
                # Deshacer elimina el último evento registrado y actualiza inmediatamente el archivo.
                evento_borrado = historial_eventos.pop()
                guardar_eventos(ruta_salida, historial_eventos)
                print(f"Deshacer: eliminado evento {evento_borrado['evento']} en frame {evento_borrado['fotograma']}")

            # En modo reproducción se avanza un fotograma mientras no se haya usado una tecla de salto manual.
            if not esta_en_pausa and tecla not in [ord("d"), ord("a"), ord("n"), ord("b")]:
                 if indice_actual < total_fotogramas - 1:
                    indice_actual += 1
                 else:
                    esta_en_pausa = True

    finally:
        captura_video.release()
        cv2.destroyAllWindows()


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""
    analizador = argparse.ArgumentParser(description="Herramienta para crear manualmente la referencia temporal de eventos IN y OUT.")

    analizador.add_argument("--video", dest="ruta_video", type=Path, required=True, help="Ruta al vídeo que se quiere anotar.")
    analizador.add_argument("--salida", dest="ruta_salida", type=Path, required=True, help="Ruta del archivo donde se guardarán los eventos frame,event.")
    analizador.add_argument("--model", dest="ruta_modelo", type=Path, required=True, help="Ruta del modelo a utilizar como apoyo visual durante la anotación.")
    analizador.add_argument("--conf", dest="confianza_modelo", type=float, default=0.15, help="Umbral de confianza utilizado para mostrar las detecciones de asistencia.")

    argumentos = analizador.parse_args()

    # Se comprueban las entradas antes de abrir la interfaz de anotación.
    if not argumentos.ruta_video.is_file():
        analizador.error(f"No existe el vídeo: {argumentos.ruta_video}")

    if argumentos.ruta_modelo is not None and not argumentos.ruta_modelo.is_file():
        analizador.error(f"No existe el modelo: {argumentos.ruta_modelo}")

    # Se evita sobrescribir por accidente una referencia temporal ya existente.
    if argumentos.ruta_salida.exists():
        analizador.error(f"El archivo de salida ya existe: {argumentos.ruta_salida}. Utiliza otra ruta para evitar sobrescribir anotaciones existentes.")

    if not 0.0 <= argumentos.confianza_modelo <= 1.0:
        analizador.error("--conf debe estar entre 0 y 1.")

    return argumentos


def main() -> None:
    """Lee los argumentos y abre la herramienta de anotación manual."""
    argumentos = obtener_argumentos()

    iniciar_anotador(
        ruta_video=argumentos.ruta_video,
        ruta_salida=argumentos.ruta_salida,
        ruta_modelo=argumentos.ruta_modelo,
        confianza_modelo=argumentos.confianza_modelo,
    )


if __name__ == "__main__":
    main()