"""Funciones de codificación y transformación de features para el modelo de predicción."""

import logging
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)

MAPA_TEMPORADA: dict[str, int] = {
    "primavera": 0,
    "verano": 1,
    "otoño": 2,
    "invierno": 3,
}

VALOR_TEMPORADA_DESCONOCIDA = -1
DIAS_POR_ANNO = 365


def encode_season(nombre_temporada: str) -> int:
    """
    Mapea el nombre de la temporada a un entero: primavera=0, verano=1, otoño=2, invierno=3.
    Retorna -1 si el valor es NaN o no reconocido.

    Parámetros:
        nombre_temporada: cadena que representa la temporada en español.

    Retorna un entero entre -1 y 3.
    """
    if pd.isna(nombre_temporada):
        return VALOR_TEMPORADA_DESCONOCIDA
    return MAPA_TEMPORADA.get(str(nombre_temporada).lower().strip(), VALOR_TEMPORADA_DESCONOCIDA)


def encode_boolean_flags(dataframe: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Convierte columnas booleanas (True/False) a enteros (1/0).
    Las columnas no listadas no se modifican. Los valores NaN se preservan.

    Parámetros:
        dataframe: DataFrame original.
        columnas: lista de nombres de columna booleanas a convertir.

    Retorna un nuevo DataFrame con las columnas convertidas.
    """
    resultado = dataframe.copy()
    for columna in columnas:
        if columna not in resultado.columns:
            logger.warning(
                "Columna booleana '%s' no encontrada en el DataFrame, se omite", columna
            )
            continue
        resultado[columna] = resultado[columna].apply(
            lambda valor: int(bool(valor)) if pd.notna(valor) else valor
        )
    return resultado


def compute_warmup_cutoff(
    dataframe: pd.DataFrame,
    columna_artista: str,
    columna_fecha: str,
    annios_warmup: int = 5,
) -> pd.DataFrame:
    """
    Añade la columna booleana 'is_warmup' marcando los primeros annios_warmup años
    de historia observada de cada artista.

    Un registro se marca is_warmup=True si se cumple alguna de estas condiciones:
      - La fecha es anterior a (min_fecha_artista + annios_warmup años).
      - La columna 'años_observados_artista' existe y su valor es <= annios_warmup.

    Parámetros:
        dataframe: DataFrame con los registros de conciertos.
        columna_artista: nombre de la columna de artista.
        columna_fecha: nombre de la columna de fecha (datetime o string parseable).
        annios_warmup: número de años a excluir desde el primer concierto observado.

    Retorna el DataFrame con la columna 'is_warmup' añadida.
    """
    resultado = dataframe.copy()
    resultado[columna_fecha] = pd.to_datetime(resultado[columna_fecha], errors="coerce")

    fecha_minima_por_artista = resultado.groupby(columna_artista)[columna_fecha].transform("min")
    cutoff_por_artista = fecha_minima_por_artista + timedelta(days=annios_warmup * DIAS_POR_ANNO)

    condicion_por_fecha = resultado[columna_fecha] < cutoff_por_artista

    if "años_observados_artista" in resultado.columns:
        condicion_por_annios = resultado["años_observados_artista"] <= annios_warmup
        resultado["is_warmup"] = condicion_por_fecha | condicion_por_annios
    else:
        resultado["is_warmup"] = condicion_por_fecha

    total_warmup = resultado["is_warmup"].sum()
    logger.info(
        "Warmup cutoff aplicado — %d de %d filas marcadas como warmup (%.1f%%)",
        total_warmup,
        len(resultado),
        (total_warmup / len(resultado) * 100) if len(resultado) > 0 else 0,
    )
    return resultado
