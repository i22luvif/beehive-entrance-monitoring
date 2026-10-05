"""
Prepara el dataset de detección a partir de secuencias revisadas manualmente.

La entrada esperada es:

    data/reviewed/
        <secuencia>/
            images/
            labels/

Cada secuencia se mantiene completa en una única partición para reducir el riesgo
de que fotogramas temporalmente próximos aparezcan simultáneamente en
entrenamiento y evaluación.

El script genera:

    datasets/bee_dataset/
        train/images/
        train/labels/
        val/images/
        val/labels/
        test/images/
        test/labels/
        data.yaml
        auditoria_dataset.csv

El reparto intenta aproximarse a las proporciones indicadas según el número de
imágenes, agrupando previamente las secuencias por su proporción de fondo.
"""


import argparse
import csv
import random
import shutil
from pathlib import Path


EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def copiar_archivo_seguro(ruta_origen: Path, ruta_destino: Path) -> None:
    """Copia un archivo sin sobrescribirlo si ya existe en el destino."""

    ruta_destino.parent.mkdir(parents=True, exist_ok=True)

    if not ruta_destino.exists():
        shutil.copy2(ruta_origen, ruta_destino)


def analizar_estadisticas_secuencia(directorio_secuencia: Path) -> dict:
    """Cuenta las imágenes positivas y de fondo de una secuencia."""

    directorio_imagenes = directorio_secuencia / "images"
    directorio_etiquetas = directorio_secuencia / "labels"

    if not directorio_imagenes.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {directorio_imagenes}")
    
    if not directorio_etiquetas.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de etiquetas: {directorio_etiquetas}"
    )

    # Solo se tienen en cuenta los formatos de imagen admitidos por el script.
    rutas_imagenes = sorted(ruta for ruta in directorio_imagenes.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_VALIDAS)

    total_imagenes = len(rutas_imagenes)
    imagenes_positivas = 0
    imagenes_fondo = 0

    for ruta_imagen in rutas_imagenes:
        ruta_etiqueta = directorio_etiquetas / f"{ruta_imagen.stem}.txt"

        # Si el archivo de etiquetas existe y tiene contenido, la imagen se considera positiva.
        if ruta_etiqueta.exists() and ruta_etiqueta.read_text(encoding="utf-8").strip():
            imagenes_positivas += 1
        else:
            imagenes_fondo += 1

    proporcion_fondo = imagenes_fondo / total_imagenes if total_imagenes > 0 else 0.0

    return {
        "nombre_secuencia": directorio_secuencia.name,
        "total_imagenes": total_imagenes,
        "imagenes_positivas": imagenes_positivas,
        "imagenes_fondo": imagenes_fondo,
        "proporcion_fondo": proporcion_fondo,
    }


def dividir_dataset_estratificado(estadisticas_secuencias: list[dict], proporcion_entrenamiento: float, proporcion_validacion: float, semilla: int) -> tuple[list[str], list[str], list[str]]:
    """Agrupa las secuencias según su proporción de fondo y las reparte completas
        entre train, val y test intentando aproximar el número objetivo de imágenes."""

    random.seed(semilla)
    proporcion_test = 1.0 - proporcion_entrenamiento - proporcion_validacion

    if proporcion_entrenamiento <= 0 or proporcion_validacion <= 0 or proporcion_test <= 0:
        raise ValueError("Las proporciones de train, val y test deben ser mayores que 0 y sumar 1.")

    # Se descartan las carpetas de secuencia que no contienen ninguna imagen.
    secuencias_validas = [estadistica for estadistica in estadisticas_secuencias if estadistica["total_imagenes"] > 0]
    
    if len(secuencias_validas) < 3:
        raise ValueError("Se necesitan al menos 3 secuencias con imágenes para particionar el dataset.")
    
    total_imagenes = sum(estadistica["total_imagenes"] for estadistica in secuencias_validas)

    objetivos_imagenes = {
        "train": total_imagenes * proporcion_entrenamiento,
        "val": total_imagenes * proporcion_validacion,
        "test": total_imagenes * proporcion_test,
    }

    asignaciones = {"train": [], "val": [], "test": []}
    imagenes_asignadas = {"train": 0, "val": 0, "test": 0}

    secuencias_alto_fondo = []
    secuencias_mixtas = []
    secuencias_bajo_fondo = []

    # Se separan las secuencias en tres grupos según la proporción de imágenes de fondo.
    for estadistica in secuencias_validas:
        proporcion_fondo = estadistica["proporcion_fondo"]

        if proporcion_fondo >= 0.70:
            secuencias_alto_fondo.append(estadistica)
        elif proporcion_fondo >= 0.10:
            secuencias_mixtas.append(estadistica)
        else:
            secuencias_bajo_fondo.append(estadistica)

    # Se cambia el orden dentro de cada grupo usando la misma semilla para poder repetir el reparto.
    random.shuffle(secuencias_alto_fondo)
    random.shuffle(secuencias_mixtas)
    random.shuffle(secuencias_bajo_fondo)

    # Se procesan primero las secuencias con más fondo, después las mixtas y por último las de menor fondo.
    for grupo in [secuencias_alto_fondo, secuencias_mixtas, secuencias_bajo_fondo]:
        for estadistica in grupo:

            # Se asigna la secuencia completa a la partición que ha alcanzado una menor proporción de su objetivo de imágenes.
            mejor_particion = min(["train", "val", "test"], key=lambda particion: imagenes_asignadas[particion] / max(objetivos_imagenes[particion], 1))

            asignaciones[mejor_particion].append(estadistica["nombre_secuencia"])
            imagenes_asignadas[mejor_particion] += estadistica["total_imagenes"]

    return asignaciones["train"], asignaciones["val"], asignaciones["test"]


def generar_subconjunto_yolo(nombre_particion: str, secuencias_asignadas: list[str], directorio_origen: Path, directorio_destino: Path) -> tuple[int, int, int]:
    """Copia las imágenes y etiquetas de una partición a la estructura usada por YOLO."""

    directorio_imagenes_salida = directorio_destino / nombre_particion / "images"
    directorio_etiquetas_salida = directorio_destino / nombre_particion / "labels"

    imagenes_procesadas = 0
    etiquetas_procesadas = 0
    imagenes_fondo = 0

    for nombre_secuencia in secuencias_asignadas:
        directorio_secuencia = directorio_origen / nombre_secuencia
        directorio_imagenes = directorio_secuencia / "images"
        directorio_etiquetas = directorio_secuencia / "labels"

        # Si falta la carpeta de imágenes se pasa a la siguiente secuencia.
        if not directorio_imagenes.exists():
            continue
            
        # Se recorren solo los archivos de imagen con extensiones válidas.
        for ruta_imagen in sorted(directorio_imagenes.iterdir()):
            if not ruta_imagen.is_file() or ruta_imagen.suffix.lower() not in EXTENSIONES_VALIDAS:
                continue

            # Se añade el nombre de la secuencia para evitar nombres repetidos entre carpetas distintas.
            prefijo_unico = f"{nombre_secuencia}__{ruta_imagen.stem}"
            ruta_imagen_destino = directorio_imagenes_salida / f"{prefijo_unico}{ruta_imagen.suffix.lower()}"
            ruta_etiqueta_origen = directorio_etiquetas / f"{ruta_imagen.stem}.txt"
            ruta_etiqueta_destino = directorio_etiquetas_salida / f"{prefijo_unico}.txt"

            copiar_archivo_seguro(ruta_imagen, ruta_imagen_destino)
            imagenes_procesadas += 1

            if ruta_etiqueta_origen.exists():
                copiar_archivo_seguro(ruta_etiqueta_origen, ruta_etiqueta_destino)
                etiquetas_procesadas += 1

                if not ruta_etiqueta_origen.read_text(encoding="utf-8").strip():
                    imagenes_fondo += 1
            else:
                # Si falta la etiqueta, se crea un archivo vacío para conservar la imagen como fondo.
                ruta_etiqueta_destino.parent.mkdir(parents=True, exist_ok=True)
                ruta_etiqueta_destino.write_text("", encoding="utf-8")
                imagenes_fondo += 1

    return imagenes_procesadas, etiquetas_procesadas, imagenes_fondo


def exportar_resumen_csv(directorio_destino: Path, estadisticas_secuencias: list[dict], secuencias_entrenamiento: list[str], secuencias_validacion: list[str], secuencias_prueba: list[str]) -> None:
    """Guarda en CSV la partición y las estadísticas de cada secuencia."""

    mapa_particiones = {}

    for secuencia in secuencias_entrenamiento:
        mapa_particiones[secuencia] = "train"

    for secuencia in secuencias_validacion:
        mapa_particiones[secuencia] = "val"

    for secuencia in secuencias_prueba:
        mapa_particiones[secuencia] = "test"

    ruta_csv = directorio_destino / "auditoria_dataset.csv"

    with ruta_csv.open("w", newline="", encoding="utf-8") as archivo_csv:
        escritor = csv.writer(archivo_csv, delimiter=";")
        escritor.writerow(["secuencia", "split_asignado", "total_imagenes", "positivas", "fondos", "ratio_fondo"])

        for estadistica in sorted(estadisticas_secuencias, key=lambda elemento: elemento["nombre_secuencia"]):
            escritor.writerow([
                estadistica["nombre_secuencia"],
                mapa_particiones.get(estadistica["nombre_secuencia"], "descartada"),
                estadistica["total_imagenes"],
                estadistica["imagenes_positivas"],
                estadistica["imagenes_fondo"],
                round(estadistica["proporcion_fondo"], 4),
            ])

    print(f"Auditoría de particiones guardada en: {ruta_csv}")


def preparar_dataset(ruta_entrada: Path, ruta_salida: Path, proporcion_entrenamiento: float, proporcion_validacion: float, semilla: int) -> None:
    """Analiza las secuencias, realiza el reparto y crea el dataset final."""

    directorio_origen = ruta_entrada
    directorio_destino = ruta_salida

    if not directorio_origen.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {directorio_origen}")


    # Si la carpeta de salida ya contiene un dataset, se detiene para no mezclar particiones de ejecuciones distintas.
    if any((directorio_destino / particion).exists() for particion in ("train", "val", "test")):
        raise FileExistsError(
            f"La carpeta de salida ya contiene un dataset: {directorio_destino}. "
            "Utiliza una carpeta vacía o elimina antes las particiones existentes."
        )
    
    
    directorio_destino.mkdir(parents=True, exist_ok=True)

    # Cada subcarpeta del directorio de entrada se trata como una secuencia independiente.
    directorios_secuencias = sorted(directorio for directorio in directorio_origen.iterdir() if directorio.is_dir())

    print(f"Secuencias detectadas: {len(directorios_secuencias)}")

    if len(directorios_secuencias) < 3:
        raise ValueError("Se necesitan al menos 3 secuencias distintas para particionar el dataset.")

    # Primero se calculan las estadísticas de todas las secuencias antes de realizar el reparto.
    estadisticas_secuencias = [analizar_estadisticas_secuencia(directorio) for directorio in directorios_secuencias]

    secuencias_entrenamiento, secuencias_validacion, secuencias_prueba = dividir_dataset_estratificado(estadisticas_secuencias, proporcion_entrenamiento, proporcion_validacion, semilla)

    print("Construyendo directorios YOLO...")

    imagenes_entrenamiento, _, fondos_entrenamiento = generar_subconjunto_yolo("train", secuencias_entrenamiento, directorio_origen, directorio_destino)
    imagenes_validacion, _, fondos_validacion = generar_subconjunto_yolo("val", secuencias_validacion, directorio_origen, directorio_destino)
    imagenes_prueba, _, fondos_prueba = generar_subconjunto_yolo("test", secuencias_prueba, directorio_origen, directorio_destino)

    print("Dataset generado:")
    print(f" - TRAIN: {imagenes_entrenamiento} imágenes ({fondos_entrenamiento} fondos)")
    print(f" - VAL:   {imagenes_validacion} imágenes ({fondos_validacion} fondos)")
    print(f" - TEST:  {imagenes_prueba} imágenes ({fondos_prueba} fondos)")

    # data.yaml indica a Ultralytics dónde están las tres particiones y qué clase contiene el dataset.
    # El campo path se guarda como ruta absoluta y debe actualizarse si el dataset se mueve posteriormente a otro equipo.
    contenido_yaml = (
        f"path: {directorio_destino.absolute()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['abeja']\n"
    )

    (directorio_destino / "data.yaml").write_text(contenido_yaml, encoding="utf-8")
    exportar_resumen_csv(directorio_destino, estadisticas_secuencias, secuencias_entrenamiento, secuencias_validacion, secuencias_prueba)


def obtener_argumentos() -> argparse.Namespace:
    """Define los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Prepara y divide por secuencias el dataset de detección en formato YOLO.")

    analizador.add_argument("--input", dest="directorio_entrada", type=Path, default=Path("data/reviewed"), help="Carpeta con las secuencias revisadas y sus directorios images/ y labels/.")
    analizador.add_argument("--output", dest="directorio_salida", type=Path, default=Path("datasets/bee_dataset"), help="Carpeta donde se generará el dataset final.")
    analizador.add_argument("--train_ratio", dest="proporcion_entrenamiento", type=float, default=0.70, help="Proporción objetivo para la partición de entrenamiento.")
    analizador.add_argument("--val_ratio", dest="proporcion_validacion", type=float, default=0.15, help="Proporción objetivo para la partición de validación.")
    analizador.add_argument("--seed", dest="semilla", type=int, default=42, help="Semilla utilizada para ordenar las secuencias de forma reproducible.")

    return analizador.parse_args()


def main() -> None:
    """Lee los argumentos y ejecuta la preparación del dataset."""

    argumentos = obtener_argumentos()

    preparar_dataset(ruta_entrada=argumentos.directorio_entrada, ruta_salida=argumentos.directorio_salida, proporcion_entrenamiento=argumentos.proporcion_entrenamiento, proporcion_validacion=argumentos.proporcion_validacion, semilla=argumentos.semilla)


if __name__ == "__main__":
    main()