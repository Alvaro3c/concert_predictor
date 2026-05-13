"""Funciones de partición temporal del dataset para entrenamiento y evaluación."""

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)


def temporal_split_by_artist(
    dataframe: pd.DataFrame,
    columna_artista: str,
    columna_fecha: str,
    ratio_train: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset de forma temporal respetando la cronología de cada artista.

    Para cada artista, ordena sus registros cronológicamente y asigna los primeros
    floor(n * ratio_train) al conjunto de train y el resto al de test. Esto garantiza
    que el modelo nunca entrena con datos del futuro de un artista.

    Los artistas con un único concierto se asignan íntegramente a train.
    Si la columna 'is_warmup' existe, las filas marcadas se excluyen antes del split.
    No mezcla registros del mismo artista entre train y test.

    Parámetros:
        dataframe: DataFrame con los registros de conciertos.
        columna_artista: nombre de la columna de artista.
        columna_fecha: nombre de la columna de fecha (datetime o string parseable).
        ratio_train: fracción de registros por artista para train (default 0.8).

    Retorna una tupla (train_df, test_df) sin solapamiento temporal por artista.
    """
    dataframe_filtrado = dataframe.copy()

    if "is_warmup" in dataframe_filtrado.columns:
        filas_antes = len(dataframe_filtrado)
        dataframe_filtrado = dataframe_filtrado[~dataframe_filtrado["is_warmup"]].drop(
            columns=["is_warmup"]
        )
        logger.info(
            "Excluidas %d filas de warmup antes del split", filas_antes - len(dataframe_filtrado)
        )

    filas_train: list[pd.DataFrame] = []
    filas_test: list[pd.DataFrame] = []

    for nombre_artista, grupo_artista in dataframe_filtrado.groupby(columna_artista, sort=False):
        grupo_ordenado = grupo_artista.sort_values(columna_fecha).reset_index(drop=True)
        total_filas_artista = len(grupo_ordenado)
        num_train = math.floor(total_filas_artista * ratio_train)

        # Artistas con un solo concierto van íntegramente a train
        if num_train == 0:
            num_train = total_filas_artista

        filas_train.append(grupo_ordenado.iloc[:num_train])
        if num_train < total_filas_artista:
            filas_test.append(grupo_ordenado.iloc[num_train:])

    columnas_df = dataframe_filtrado.columns.tolist()
    train_df = (
        pd.concat(filas_train, ignore_index=True) if filas_train else pd.DataFrame(columns=columnas_df)
    )
    test_df = (
        pd.concat(filas_test, ignore_index=True) if filas_test else pd.DataFrame(columns=columnas_df)
    )

    logger.info(
        "Split temporal completado — Train: %d filas | Test: %d filas | Ratio efectivo: %.2f",
        len(train_df),
        len(test_df),
        len(train_df) / (len(train_df) + len(test_df)) if (len(train_df) + len(test_df)) > 0 else 0,
    )
    return train_df, test_df
