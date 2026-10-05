"""
Extrae fotogramas de vídeos y genera preanotaciones automáticas con YOLO.

Por defecto, los vídeos se buscan de forma recursiva dentro de data/videos_originales/.
Se espera una organización del tipo:

    data/videos_originales/<dia>/<video>.mp4

Para cada vídeo se genera:

    data/preannotations/<dia>/<video>/
        images/
        labels/
        confidences/

Las etiquetas de labels/ utilizan el formato YOLO normalizado. La carpeta
confidences/ conserva además la confianza de cada predicción para facilitar
la revisión posterior.
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

def preanotar_video(ruta_video: Path, modelo: YOLO, directorio_salida: Path, salto_fotogramas: int, umbral_confianza: float) -> None:
    """Extrae fotogramas de un vídeo y guarda las preanotaciones generadas por YOLO."""

    captura_video = cv2.VideoCapture(str(ruta_video))

    if not captura_video.isOpened():
        print(f"[ERROR] No se pudo abrir el vídeo: {ruta_video}")
        return

    nombre_video = ruta_video.stem
    nombre_carpeta_padre = ruta_video.parent.name

    # Estructura: directorio_salida / carpeta_padre / nombre_video / {images, labels, confidences}
    directorio_secuencia = directorio_salida / nombre_carpeta_padre / nombre_video
    directorio_imagenes = directorio_secuencia / "images"
    directorio_etiquetas = directorio_secuencia / "labels"
    directorio_confianzas = directorio_secuencia / "confidences"

    # Se crean las carpetas necesarias para guardar las salidas de cada vídeo.
    directorio_imagenes.mkdir(parents=True, exist_ok=True)
    directorio_etiquetas.mkdir(parents=True, exist_ok=True)
    directorio_confianzas.mkdir(parents=True, exist_ok=True)

    # El índice comienza en 0, por lo que con --skip 15 se procesan los fotogramas 0, 15, 30, 45, ...
    indice_fotograma = 0
    fotogramas_guardados = 0

    print(f"Procesando {nombre_video}: 1 fotograma de cada {salto_fotogramas}, confianza mínima {umbral_confianza:.2f}")

    try:
        while captura_video.isOpened():
            lectura_correcta, fotograma = captura_video.read()

            if not lectura_correcta:
                break

            # Solo se procesa un fotograma cuando cumple el intervalo de muestreo indicado.
            if indice_fotograma % salto_fotogramas == 0:

                # Se ejecuta el modelo usando el umbral mínimo de confianza indicado.
                resultados = modelo(fotograma, conf=umbral_confianza, verbose=False)

                # Se añade el número del fotograma original al nombre para poder localizarlo después en el vídeo.
                nombre_fotograma = f"{nombre_video}_{indice_fotograma:05d}"

                ruta_imagen = directorio_imagenes / f"{nombre_fotograma}.jpg"
                ruta_etiqueta = directorio_etiquetas / f"{nombre_fotograma}.txt"
                ruta_confianza = directorio_confianzas / f"{nombre_fotograma}.txt"

                # Se guarda el fotograma original antes de escribir sus anotaciones.
                if not cv2.imwrite(str(ruta_imagen), fotograma):
                    raise OSError(f"No se pudo guardar la imagen: {ruta_imagen}")

                cajas = resultados[0].boxes

                # Los archivos se crean aunque no haya detecciones. En ese caso quedan vacíos.
                with ruta_etiqueta.open("w", encoding="utf-8") as archivo_etiqueta, ruta_confianza.open("w", encoding="utf-8") as archivo_confianza:

                    if len(cajas) > 0:

                        # xywhn contiene el centro, ancho y alto de cada caja normalizados entre 0 y 1.
                        coordenadas = cajas.xywhn.cpu().numpy()
                        clases = cajas.cls.cpu().numpy()
                        confianzas = cajas.conf.cpu().numpy()

                        for clase, coordenada, confianza in zip(clases, coordenadas, confianzas):

                            # Formato de anotación YOLO: clase x_centro y_centro ancho alto.
                            linea_yolo = f"{int(clase)} {coordenada[0]:.6f} {coordenada[1]:.6f} {coordenada[2]:.6f} {coordenada[3]:.6f}\n"
                            archivo_etiqueta.write(linea_yolo)

                            # La confianza se guarda aparte, ya que no forma parte de las etiquetas YOLO.
                            linea_confianza = f"{int(clase)} {coordenada[0]:.6f} {coordenada[1]:.6f} {coordenada[2]:.6f} {coordenada[3]:.6f} {confianza:.4f}\n"
                            archivo_confianza.write(linea_confianza)

                fotogramas_guardados += 1

            indice_fotograma += 1

    finally:
        # Se libera el vídeo aunque ocurra algún error durante el procesamiento.
        captura_video.release()

    print(f"Finalizado {nombre_video}: {fotogramas_guardados} imágenes guardadas.")


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Extrae fotogramas de vídeos MP4 y genera preanotaciones automáticas en formato YOLO.")

    analizador.add_argument("--input_dir", dest="directorio_entrada", type=Path, default=Path("data/videos_originales"), help="Carpeta raíz que contiene los vídeos originales.")
    analizador.add_argument("--model", dest="ruta_modelo", type=Path, required=True, help="Ruta al checkpoint de YOLO utilizado para generar las preanotaciones.")
    analizador.add_argument("--output", dest="directorio_salida", type=Path, default=Path("data/preannotations"), help="Carpeta donde se guardarán las preanotaciones.")
    analizador.add_argument("--skip", dest="salto_fotogramas", type=int, default=15, help="Intervalo utilizado para seleccionar los fotogramas.")
    analizador.add_argument("--conf", dest="umbral_confianza", type=float, default=0.3, help="Umbral mínimo de confianza para aceptar una detección.")

    argumentos = analizador.parse_args()

    # Se comprueban los parámetros antes de comenzar a procesar los vídeos.
    if argumentos.salto_fotogramas < 1:
        analizador.error("--skip debe ser un entero mayor o igual que 1.")

    if not 0.0 <= argumentos.umbral_confianza <= 1.0:
        analizador.error("--conf debe estar entre 0 y 1.")

    if not argumentos.directorio_entrada.is_dir():
        analizador.error(f"No existe la carpeta de entrada: {argumentos.directorio_entrada}")
        
    if not argumentos.ruta_modelo.is_file():
        analizador.error(f"No existe el checkpoint indicado: {argumentos.ruta_modelo}")
    return argumentos


def main() -> None:
    """Busca los vídeos y ejecuta la preanotación de cada uno."""

    argumentos = obtener_argumentos()

    # Se buscan de forma recursiva todos los vídeos MP4 dentro de la carpeta de entrada.
    videos = sorted(argumentos.directorio_entrada.rglob("*.mp4"))

    if not videos:
        print(f"No se encontraron vídeos .mp4 en: {argumentos.directorio_entrada}")
        return

    print(f"Vídeos encontrados: {len(videos)}")
    print(f"Cargando modelo: {argumentos.ruta_modelo}")

    # El modelo se carga una sola vez y después se reutiliza para todos los vídeos.
    modelo = YOLO(str(argumentos.ruta_modelo))

    # Cada vídeo se procesa de forma independiente y guarda su propia carpeta de resultados.
    for indice, ruta_video in enumerate(videos, start=1):
        print(f"[{indice}/{len(videos)}] {ruta_video}")

        preanotar_video(
            ruta_video=ruta_video,
            modelo=modelo,
            directorio_salida=argumentos.directorio_salida,
            salto_fotogramas=argumentos.salto_fotogramas,
            umbral_confianza=argumentos.umbral_confianza
        )

    print("Preanotación finalizada.")


if __name__ == "__main__":
    main()