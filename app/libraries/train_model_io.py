"""Funciones de persistencia y carga del modelo entrenado y sus metadatos."""

import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

NIVEL_COMPRESION = 3


def save_model(modelo, ruta_salida: Path | str) -> None:
    """
    Persiste el modelo entrenado en disco usando joblib con compresión nivel 3.

    Parámetros:
        modelo: modelo LightGBM entrenado (LGBMRegressor).
        ruta_salida: ruta del fichero .joblib de destino.
    """
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta, compress=NIVEL_COMPRESION)
    logger.info("Modelo guardado en %s", ruta)


def load_model(ruta_modelo: Path | str):
    """
    Carga un modelo previamente serializado con joblib desde disco.

    Parámetros:
        ruta_modelo: ruta del fichero .joblib del modelo.

    Retorna el modelo deserializado listo para inferencia.
    Lanza FileNotFoundError si el fichero no existe.
    """
    ruta = Path(ruta_modelo)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el modelo en la ruta: {ruta}")

    modelo = joblib.load(ruta)
    logger.info("Modelo cargado desde %s", ruta)
    return modelo


def save_feature_importance(
    modelo,
    nombres_features: list[str],
    ruta_salida: Path | str,
) -> None:
    """
    Guarda un CSV con nombre e importancia de cada feature, ordenado de mayor a menor.

    Parámetros:
        modelo: modelo LightGBM entrenado con atributo feature_importances_.
        nombres_features: lista de nombres de features en el mismo orden que el modelo.
        ruta_salida: ruta del CSV de salida.
    """
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    dataframe_importancia = pd.DataFrame(
        {"feature": nombres_features, "importance": modelo.feature_importances_}
    ).sort_values("importance", ascending=False)

    dataframe_importancia.to_csv(ruta, index=False)
    logger.info(
        "Feature importance guardada en %s (top feature: %s)",
        ruta,
        dataframe_importancia.iloc[0]["feature"],
    )
