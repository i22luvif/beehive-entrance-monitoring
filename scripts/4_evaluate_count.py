"""
Evalúa temporalmente los eventos IN y OUT generados por los trackers.

Para cada vídeo, los eventos predichos se comparan con el Ground Truth
mediante una tolerancia temporal configurable. IN y OUT se evalúan de forma
independiente.

El emparejamiento es uno a uno, cronológico y voraz: para cada evento de
referencia se selecciona la predicción disponible del mismo tipo más próxima
dentro de la tolerancia permitida.

Los valores TP, FP y FN se acumulan sobre todos los vídeos antes de calcular
la Precision y el Recall globales.

También se calcula el error absoluto global de conteo como:

    abs(numero_eventos_gt - numero_eventos_predichos)

El resultado se almacena en:

    results/tracker_comparison/metricas_globales_tfg.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def evaluar_eventos_temporales(referencia: pd.DataFrame, predicciones: pd.DataFrame, tolerancia_fotogramas: int = 30) -> dict:
    """
        Compara los eventos predichos con la referencia temporal.

        IN y OUT se procesan de forma independiente. Los eventos de referencia se
        recorren cronológicamente y cada uno se asocia, como máximo, con una predicción
        del mismo tipo situada dentro de la tolerancia ±tolerancia_fotogramas.

        Si hay varias predicciones disponibles, se selecciona la temporalmente más próxima.
    """
    
    metricas = {
        "IN": {"TP": 0, "FP": 0, "FN": 0},
        "OUT": {"TP": 0, "FP": 0, "FN": 0},
    }

    # Guarda las predicciones que ya han sido emparejadas para no utilizarlas más de una vez.
    indices_predicciones_usados = set()

    # Los eventos IN y OUT se evalúan por separado.
    for tipo_evento in ["IN", "OUT"]:
        eventos_referencia = referencia[referencia["event"] == tipo_evento].sort_values("frame")
        eventos_predichos = predicciones[predicciones["event"] == tipo_evento].sort_values("frame")

        # Se recorren cronológicamente todos los eventos de la referencia.
        for _, fila_referencia in eventos_referencia.iterrows():
            fotograma_referencia = fila_referencia["frame"]

            # Se buscan predicciones dentro de la tolerancia que todavía no hayan sido utilizadas.
            candidatos = eventos_predichos[
                (eventos_predichos["frame"] >= fotograma_referencia - tolerancia_fotogramas)
                & (eventos_predichos["frame"] <= fotograma_referencia + tolerancia_fotogramas)
                & (~eventos_predichos.index.isin(indices_predicciones_usados))
            ]

            if not candidatos.empty:
                # Si existen varios candidatos, se utiliza el más cercano al fotograma de referencia.
                indice_mas_cercano = (candidatos["frame"] - fotograma_referencia).abs().idxmin()
                indices_predicciones_usados.add(indice_mas_cercano)
                metricas[tipo_evento]["TP"] += 1
            else:
                # Si no hay ninguna predicción válida, el evento real se considera un falso negativo.
                metricas[tipo_evento]["FN"] += 1

        # Las predicciones que no han sido emparejadas se consideran falsos positivos.
        indices_tipo_evento = set(eventos_predichos.index)
        predicciones_emparejadas = len(indices_tipo_evento.intersection(indices_predicciones_usados))
        metricas[tipo_evento]["FP"] = len(eventos_predichos) - predicciones_emparejadas

    # También se guardan los conteos totales para calcular después el error absoluto.
    metricas["IN"]["GT_TOTAL"] = len(referencia[referencia["event"] == "IN"])
    metricas["IN"]["TRK_TOTAL"] = len(predicciones[predicciones["event"] == "IN"])
    metricas["OUT"]["GT_TOTAL"] = len(referencia[referencia["event"] == "OUT"])
    metricas["OUT"]["TRK_TOTAL"] = len(predicciones[predicciones["event"] == "OUT"])

    return metricas


def evaluar_dataset_completo(directorio_gt: Path, directorio_benchmark: Path, directorio_salida: Path, tolerancia: int) -> None:
    """Evalúa todos los trackers frente al Ground Truth y genera las métricas globales."""

    print("Evaluación temporal global")
    print(f"Ground Truth: {directorio_gt}")
    print(f"Resultados de trackers: {directorio_benchmark}")
    print(f"Salida: {directorio_salida}")
    print(f"Tolerancia: ±{tolerancia} fotogramas")

    directorio_salida.mkdir(parents=True, exist_ok=True)

    # El Ground Truth puede estar guardado en archivos TXT o CSV con el formato frame,event.
    archivos_gt = sorted(list(directorio_gt.glob("*.txt")) + list(directorio_gt.glob("*.csv")))

    if not archivos_gt:
         raise FileNotFoundError(f"No se encontraron archivos de Ground Truth en: {directorio_gt}")

    # Aquí se van acumulando las métricas de todos los vídeos para cada tracker.
    metricas_globales = {}

    for archivo_gt in archivos_gt:
        nombre_video = archivo_gt.stem
        referencia = pd.read_csv(archivo_gt)

       
        # Se buscan todos los resultados de trackers correspondientes al vídeo actual.
        archivos_tracker = sorted(directorio_benchmark.glob(f"results_*_{nombre_video}.csv"))
        
        if not archivos_tracker:
                    raise FileNotFoundError(f"No se encontraron resultados de trackers para el vídeo: {nombre_video}")

        for archivo_tracker in archivos_tracker:

            # El nombre del tracker se obtiene del patrón results_<tracker>_<video>.csv.
            partes_nombre = archivo_tracker.stem.split("_")
            nombre_tracker = partes_nombre[1].upper()

            predicciones = pd.read_csv(archivo_tracker)

            # Primero se calculan las métricas del tracker para este vídeo concreto.
            metricas_video = evaluar_eventos_temporales(referencia, predicciones, tolerancia)

            # La primera vez que aparece un tracker se crea su acumulador global.
            if nombre_tracker not in metricas_globales:
                metricas_globales[nombre_tracker] = {
                    "IN": {"TP": 0, "FP": 0, "FN": 0, "GT_TOTAL": 0, "TRK_TOTAL": 0},
                    "OUT": {"TP": 0, "FP": 0, "FN": 0, "GT_TOTAL": 0, "TRK_TOTAL": 0},
                }

            # Se suman los resultados de cada vídeo antes de calcular las métricas finales.
            for tipo_evento in ["IN", "OUT"]:
                for nombre_metrica in ["TP", "FP", "FN", "GT_TOTAL", "TRK_TOTAL"]:
                    metricas_globales[nombre_tracker][tipo_evento][nombre_metrica] += metricas_video[tipo_evento][nombre_metrica]

    resultados = []

    # Se calculan Precision, Recall y error absoluto usando los valores acumulados de todos los vídeos.
    for nombre_tracker in sorted(metricas_globales):
        datos = metricas_globales[nombre_tracker]

        tp_in = datos["IN"]["TP"]
        fp_in = datos["IN"]["FP"]
        fn_in = datos["IN"]["FN"]

        precision_in = tp_in / (tp_in + fp_in) if (tp_in + fp_in) > 0 else 0
        recall_in = tp_in / (tp_in + fn_in) if (tp_in + fn_in) > 0 else 0
        
        # El error absoluto se calcula sobre los conteos globales acumulados.
        # No corresponde a una media de errores por vídeo (MAE).
        error_absoluto_in = abs(datos["IN"]["GT_TOTAL"] - datos["IN"]["TRK_TOTAL"])

        tp_out = datos["OUT"]["TP"]
        fp_out = datos["OUT"]["FP"]
        fn_out = datos["OUT"]["FN"]

        precision_out = tp_out / (tp_out + fp_out) if (tp_out + fp_out) > 0 else 0
        recall_out = tp_out / (tp_out + fn_out) if (tp_out + fn_out) > 0 else 0
        error_absoluto_out = abs(datos["OUT"]["GT_TOTAL"] - datos["OUT"]["TRK_TOTAL"])

        # Se mantienen los nombres originales de las columnas del CSV de resultados.
        resultados.append({
            "Tracker": nombre_tracker,
            "TP_IN": tp_in,
            "FP_IN": fp_in,
            "FN_IN": fn_in,
            "Prec_IN": precision_in,
            "Rec_IN": recall_in,
            "ABS_ERROR_IN": error_absoluto_in,
            "TP_OUT": tp_out,
            "FP_OUT": fp_out,
            "FN_OUT": fn_out,
            "Prec_OUT": precision_out,
            "Rec_OUT": recall_out,
            "ABS_ERROR_OUT": error_absoluto_out,
            "GT_Eventos_Total": datos["IN"]["GT_TOTAL"] + datos["OUT"]["GT_TOTAL"],
            "TRK_Eventos_Total": datos["IN"]["TRK_TOTAL"] + datos["OUT"]["TRK_TOTAL"],
        })

    tabla_resultados = pd.DataFrame(resultados)

    print("=" * 85)
    print("RESULTADOS GLOBALES DEL BENCHMARK DE CONTEO")
    print("=" * 85)

    # Precision y Recall se muestran como porcentajes únicamente en la salida por consola.
    formato = {
        "Prec_IN": "{:.1%}".format,
        "Rec_IN": "{:.1%}".format,
        "Prec_OUT": "{:.1%}".format,
        "Rec_OUT": "{:.1%}".format,
    }

    print(tabla_resultados.to_string(index=False, formatters=formato))
    print("=" * 85)

    # El CSV conserva los valores numéricos originales para poder utilizarlos en análisis posteriores.
    ruta_resultados = directorio_salida / "metricas_globales_tfg.csv"
    # Una nueva ejecución sobrescribe el CSV global existente dentro de la carpeta de salida.
    tabla_resultados.to_csv(ruta_resultados, index=False)

    print(f"Resultados guardados en: {ruta_resultados}")


def obtener_argumentos() -> argparse.Namespace:
    """Define y valida los argumentos recibidos por línea de comandos."""

    analizador = argparse.ArgumentParser(description="Evalúa temporalmente los eventos generados por los trackers frente al Ground Truth.")

    analizador.add_argument("--gt_dir", dest="directorio_gt", type=Path, default=Path("data/gt"), help="Carpeta con los archivos de Ground Truth.")
    analizador.add_argument("--benchmark_dir", dest="directorio_benchmark", type=Path, default=Path("results/benchmark"), help="Carpeta con los CSV de eventos generados por los trackers.")
    analizador.add_argument("--output_dir", dest="directorio_salida", type=Path, default=Path("results/tracker_comparison"), help="Carpeta donde se guardarán las métricas globales.")
    analizador.add_argument("--tolerancia", type=int, default=30, help="Tolerancia temporal en fotogramas para emparejar eventos.")

    argumentos = analizador.parse_args()

    # Se comprueban las rutas y la tolerancia antes de comenzar la evaluación.
    if not argumentos.directorio_gt.is_dir():
        analizador.error(f"No existe la carpeta de Ground Truth: {argumentos.directorio_gt}")

    if not argumentos.directorio_benchmark.is_dir():
        analizador.error(f"No existe la carpeta de resultados: {argumentos.directorio_benchmark}")

    if argumentos.tolerancia < 0:
        analizador.error("--tolerancia debe ser mayor o igual que 0.")

    return argumentos


def main() -> None:
    """Lee los argumentos y ejecuta la evaluación temporal completa."""

    argumentos = obtener_argumentos()

    evaluar_dataset_completo(
        directorio_gt=argumentos.directorio_gt,
        directorio_benchmark=argumentos.directorio_benchmark,
        directorio_salida=argumentos.directorio_salida,
        tolerancia=argumentos.tolerancia,
    )


if __name__ == "__main__":
    main()