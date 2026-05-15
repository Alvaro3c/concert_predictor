import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.libraries.date_utils import parsear_fecha
from app.libraries.window_utils import (
    conciertos_en_ventana,
    dias_desde_ultimo_concierto,
    calcular_tendencia_actividad,
    calcular_ratio_expansion_artista,
    años_observados_artista,
)

RUTA_FUENTE = "data/interim/normalized_concerts.jsonl"
RUTA_DESTINO = "data/processed/conciertos_enriquecido.jsonl"

logger = logging.getLogger(__name__)


def calcular_features_artista_global(
    registros_previos_artista: list[dict],
    fecha_actual: datetime,
) -> dict:
    """
    Calcula las features globales de actividad de un artista para un registro dado.

    Recibe únicamente los registros del mismo artista con fecha estrictamente anterior
    a fecha_actual, garantizando la ausencia de data leakage.

    Parámetros:
        registros_previos_artista: lista de dicts con al menos el campo "fecha" (str YYYY-MM-DD).
        fecha_actual: fecha del registro en curso como objeto datetime.

    Devuelve un dict con las siguientes keys:
        conciertos_ultimo_año, conciertos_ultimos_3_años, conciertos_ultimos_5_años,
        dias_desde_ultimo_concierto_global, tendencia_actividad, años_observados_artista.
    Cualquier feature que no pueda calcularse por falta de datos se devuelve como None.
    """
    fechas_previas = [
        parsear_fecha(registro.get("fecha"))
        for registro in registros_previos_artista
    ]
    fechas_previas = [fecha for fecha in fechas_previas if fecha is not None]

    return {
        "conciertos_ultimo_año": conciertos_en_ventana(fechas_previas, fecha_actual, 365),
        "conciertos_ultimos_3_años": conciertos_en_ventana(fechas_previas, fecha_actual, 1095),
        "conciertos_ultimos_5_años": conciertos_en_ventana(fechas_previas, fecha_actual, 1825),
        "dias_desde_ultimo_concierto_global": dias_desde_ultimo_concierto(fechas_previas, fecha_actual),
        "tendencia_actividad": calcular_tendencia_actividad(fechas_previas, fecha_actual),
        "ratio_expansion_artista": calcular_ratio_expansion_artista(fechas_previas, fecha_actual),
        "años_observados_artista": años_observados_artista(fechas_previas, fecha_actual),
    }


def _convertir_nan_a_none(registro: dict) -> dict:
    """Reemplaza valores NaN de pandas/numpy por None para serialización JSON correcta."""
    return {
        clave: (None if isinstance(valor, float) and pd.isna(valor) else valor)
        for clave, valor in registro.items()
    }


def enriquecer_conciertos() -> dict:
    """
    Pipeline completo de enriquecimiento con artist global features.

    1. Carga el JSONL normalizado de data/interim/normalized_concerts.jsonl en un DataFrame.
    2. Agrupa por artista para evitar iterar el dataset completo por cada registro.
    3. Para cada registro, calcula las features usando solo los conciertos del mismo artista
       con fecha estrictamente anterior (sin data leakage).
    4. Guarda el resultado enriquecido en data/processed/conciertos_enriquecido.jsonl.

    Devuelve un dict resumen con el total de registros procesados y la ruta de destino.
    """
    ruta_fuente = Path(RUTA_FUENTE)
    if not ruta_fuente.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {RUTA_FUENTE}")

    dataframe = pd.read_json(ruta_fuente, lines=True, dtype=False)
    total_cargados = len(dataframe)
    logger.info("Cargados %d registros desde %s", total_cargados, RUTA_FUENTE)

    # Parsear la columna de fecha a datetime para poder comparar y ordenar
    dataframe["fecha_dt"] = pd.to_datetime(dataframe["fecha"], format="%Y-%m-%d", errors="coerce")

    # Ordenar globalmente por artista y fecha antes de agrupar
    dataframe = dataframe.sort_values(["artista", "fecha_dt"]).reset_index(drop=True)

    # Columnas originales sin la columna auxiliar de datetime
    columnas_originales = [columna for columna in dataframe.columns if columna != "fecha_dt"]

    resultados = []
    total_sin_fecha_valida = 0
    total_artistas = dataframe["artista"].nunique()
    artistas_procesados = 0
    print(f"[1/5 Global] Iniciando — {total_cargados} registros, {total_artistas} artistas", flush=True)

    # Agrupar por artista para acceder solo a los conciertos del mismo artista en cada iteración
    for nombre_artista, grupo_artista in dataframe.groupby("artista", sort=False):
        grupo_artista = grupo_artista.reset_index(drop=True)
        registros_artista = grupo_artista.to_dict(orient="records")
        artistas_procesados += 1
        if artistas_procesados % 5 == 0 or artistas_procesados == total_artistas:
            print(f"[1/5 Global]  {artistas_procesados}/{total_artistas} artistas — {nombre_artista} ({len(registros_artista)} conciertos) — {len(resultados)} registros acumulados", flush=True)

        for indice, registro_actual in enumerate(registros_artista):
            fecha_dt_actual = registro_actual.get("fecha_dt")

            registro_enriquecido = {
                clave: registro_actual[clave]
                for clave in columnas_originales
                if clave in registro_actual
            }

            # Sin fecha válida no se pueden calcular features
            if pd.isna(fecha_dt_actual):
                for nombre_feature in (
                    "conciertos_ultimo_año",
                    "conciertos_ultimos_3_años",
                    "conciertos_ultimos_5_años",
                    "dias_desde_ultimo_concierto_global",
                    "tendencia_actividad",
                    "ratio_expansion_artista",
                    "años_observados_artista",
                ):
                    registro_enriquecido[nombre_feature] = None
                resultados.append(registro_enriquecido)
                total_sin_fecha_valida += 1
                continue

            fecha_actual_dt = fecha_dt_actual.to_pydatetime()

            # Solo registros del mismo artista con fecha estrictamente anterior (sin data leakage)
            registros_previos = [
                {clave: r[clave] for clave in columnas_originales if clave in r}
                for r in registros_artista[:indice]
                if not pd.isna(r.get("fecha_dt")) and r["fecha_dt"] < fecha_dt_actual
            ]

            features = calcular_features_artista_global(registros_previos, fecha_actual_dt)
            registro_enriquecido.update(features)
            resultados.append(registro_enriquecido)

    print(f"[1/5 Global]  Todos los artistas procesados. Escribiendo fichero...", flush=True)
    ruta_destino = Path(RUTA_DESTINO)
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_destino, "w", encoding="utf-8") as archivo_salida:
        for registro in resultados:
            linea = _convertir_nan_a_none(registro)
            archivo_salida.write(json.dumps(linea, ensure_ascii=False) + "\n")

    print(f"[1/5 Global] Completado — {len(resultados)} registros escritos en {RUTA_DESTINO}", flush=True)
    resumen = {
        "total_procesados": len(resultados),
        "sin_fecha_valida": total_sin_fecha_valida,
        "archivo_destino": RUTA_DESTINO,
    }

    logger.info(
        "Enriquecimiento completado — procesados: %d | sin fecha válida: %d",
        len(resultados),
        total_sin_fecha_valida,
    )

    return resumen
