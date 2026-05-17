"""Funciones de persistencia y carga del modelo entrenado y sus metadatos."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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


NOMBRE_ENCODER_ARTISTA = "artista_encoder.joblib"
NOMBRE_ENCODER_PAIS = "pais_encoder.joblib"


def save_encoders(
    encoder_artista: LabelEncoder,
    encoder_pais: LabelEncoder,
    ruta_directorio: Path | str,
) -> None:
    """
    Persiste los LabelEncoders de artista y país en disco para reutilizarlos en inferencia.

    Parámetros:
        encoder_artista: LabelEncoder ya entrenado sobre la columna artista.
        encoder_pais: LabelEncoder ya entrenado sobre la columna pais.
        ruta_directorio: directorio donde guardar los ficheros .joblib.
    """
    directorio = Path(ruta_directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    ruta_artista = directorio / NOMBRE_ENCODER_ARTISTA
    ruta_pais = directorio / NOMBRE_ENCODER_PAIS

    joblib.dump(encoder_artista, ruta_artista, compress=NIVEL_COMPRESION)
    joblib.dump(encoder_pais, ruta_pais, compress=NIVEL_COMPRESION)

    logger.info(
        "Encoders guardados — artista: %d clases, pais: %d clases",
        len(encoder_artista.classes_),
        len(encoder_pais.classes_),
    )


def load_encoders(ruta_directorio: Path | str) -> tuple[LabelEncoder, LabelEncoder]:
    """
    Carga los LabelEncoders de artista y país desde disco.

    Parámetros:
        ruta_directorio: directorio donde están los ficheros artista_encoder.joblib
                         y pais_encoder.joblib.

    Retorna una tupla (encoder_artista, encoder_pais).
    Lanza FileNotFoundError si alguno de los ficheros no existe.
    """
    directorio = Path(ruta_directorio)
    ruta_artista = directorio / NOMBRE_ENCODER_ARTISTA
    ruta_pais = directorio / NOMBRE_ENCODER_PAIS

    if not ruta_artista.exists():
        raise FileNotFoundError(f"No se encontró el encoder de artista en: {ruta_artista}")
    if not ruta_pais.exists():
        raise FileNotFoundError(f"No se encontró el encoder de país en: {ruta_pais}")

    encoder_artista = joblib.load(ruta_artista)
    encoder_pais = joblib.load(ruta_pais)

    logger.info(
        "Encoders cargados — artista: %d clases, pais: %d clases",
        len(encoder_artista.classes_),
        len(encoder_pais.classes_),
    )
    return encoder_artista, encoder_pais


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
