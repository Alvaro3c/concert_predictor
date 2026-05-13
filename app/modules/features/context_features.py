import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.libraries.date_utils import (
    parsear_fecha,
    obtener_estacion,
    calcular_es_post_covid,
    calcular_es_periodo_covid,
)

RUTA_ENRIQUECIDO = "data/processed/conciertos_enriquecido.jsonl"

logger = logging.getLogger(__name__)


def calcular_features_contexto(fecha: datetime | None) -> dict:
    """
    Calcula las features de contexto temporal para un registro dado.

    Solo depende de la fecha del registro — no requiere historial ni agrupación previa.
    Si la fecha es None devuelve None en todas las features.

    Parámetros:
        fecha: objeto datetime con la fecha del registro, o None si no es válida.

    Devuelve un dict con las keys:
        estacion, es_post_covid, es_periodo_covid.
    """
    if fecha is None:
        return {
            "estacion": None,
            "es_post_covid": None,
            "es_periodo_covid": None,
        }

    return {
        "estacion": obtener_estacion(fecha),
        "es_post_covid": calcular_es_post_covid(fecha),
        "es_periodo_covid": calcular_es_periodo_covid(fecha),
    }


def _convertir_nan_a_none(registro: dict) -> dict:
    """Reemplaza valores NaN de pandas/numpy por None para serialización JSON correcta."""
    return {
        clave: (None if isinstance(valor, float) and pd.isna(valor) else valor)
        for clave, valor in registro.items()
    }


def enriquecer_context_features() -> dict:
    """
    Pipeline de enriquecimiento con context features.

    Lee data/processed/conciertos_enriquecido.jsonl, añade las context features
    y sobreescribe el mismo fichero. Es el último paso del pipeline de features:
    el JSONL resultante es el dataset final listo para entrenar el modelo.

    No requiere agrupación previa — las features se calculan registro a registro
    a partir únicamente de la fecha de cada uno.

    Devuelve un dict resumen con total de registros procesados y ruta de destino.
    """
    ruta_enriquecido = Path(RUTA_ENRIQUECIDO)
    if not ruta_enriquecido.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {RUTA_ENRIQUECIDO}")

    dataframe = pd.read_json(ruta_enriquecido, lines=True, dtype=False)
    logger.info("Cargados %d registros desde %s", len(dataframe), RUTA_ENRIQUECIDO)
    print(f"[5/5 Context] Iniciando — {len(dataframe)} registros", flush=True)

    total_sin_fecha_valida = 0
    resultados = []

    for registro in dataframe.to_dict(orient="records"):
        fecha = parsear_fecha(registro.get("fecha"))

        if fecha is None:
            total_sin_fecha_valida += 1

        features = calcular_features_contexto(fecha)
        registro.update(features)
        resultados.append(_convertir_nan_a_none(registro))

    print(f"[5/5 Context]  Escribiendo fichero...", flush=True)
    with open(ruta_enriquecido, "w", encoding="utf-8") as archivo_salida:
        for registro in resultados:
            archivo_salida.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"[5/5 Context] Completado — {len(resultados)} registros escritos. Pipeline finalizado.", flush=True)
    resumen = {
        "total_procesados": len(resultados),
        "sin_fecha_valida": total_sin_fecha_valida,
        "archivo_destino": RUTA_ENRIQUECIDO,
    }

    logger.info(
        "Context features completadas — procesados: %d | sin fecha válida: %d",
        len(resultados),
        total_sin_fecha_valida,
    )

    return resumen
