"""Funciones de validación de datos de entrada para el pipeline de entrenamiento."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

COLUMNAS_MINIMAS = ["artista", "pais", "fecha"]

COLUMNAS_FEATURES = [
    "conciertos_ultimo_año",
    "conciertos_ultimos_3_años",
    "conciertos_ultimos_5_años",
    "dias_desde_ultimo_concierto_global",
    "tendencia_actividad",
    "ratio_expansion_artista",
    "años_observados_artista",
    "conciertos_artista_ultimos_30_dias",
    "conciertos_artista_ultimos_90_dias",
    "en_gira_activa",
    "dias_desde_concierto_anterior",
    "paises_distintos_ultimos_30_dias",
    "visitas_previas_al_pais",
    "es_primera_visita_observada",
    "dias_desde_ultima_visita_pais",
    "gap_medio_pais",
    "gap_mediano_pais",
    "gap_std_pais",
    "gap_min_pais",
    "gap_max_pais",
    "ultimo_gap_pais",
    "tendencia_gap_pais",
    "gap_coeficiente_variacion_pais",
    "ratio_visitas_misma_estacion_pais",
    "concentracion_estacional_pais",
    "proporcion_visitas_pais",
    "rank_pais_para_artista",
    "paises_unicos_visitados_total",
    "paises_unicos_visitados_ultimos_3_años",
    "paises_unicos_visitados_ultimos_5_años",
    "ciudades_unicas_ultimos_5_años",
    "ratio_paises_conciertos_5y",
    "conciertos_totales_pais_previos",
    "gap_medio_global_pais",
    "estacion",
    "es_post_covid",
    "es_periodo_covid",
    "artista",
    "pais",
]

PATRONES_LEAKAGE = ["shift", "next", "futuro", "siguiente", "proximo"]

UMBRAL_NAT = 0.05


def validate_required_columns(dataframe: pd.DataFrame, columnas_requeridas: list[str]) -> bool:
    """
    Verifica que el DataFrame contenga todas las columnas requeridas.

    Parámetros:
        dataframe: DataFrame a validar.
        columnas_requeridas: lista de nombres de columna que deben estar presentes.

    Retorna True si todas las columnas existen, False en caso contrario.
    """
    columnas_faltantes = [col for col in columnas_requeridas if col not in dataframe.columns]
    if columnas_faltantes:
        logger.warning("Columnas requeridas no encontradas: %s", columnas_faltantes)
        return False
    return True


def validate_date_column(dataframe: pd.DataFrame, columna_fecha: str) -> bool:
    """
    Verifica que la columna de fecha sea parseable como datetime y no contenga
    más de un 5% de valores NaT.

    Parámetros:
        dataframe: DataFrame a validar.
        columna_fecha: nombre de la columna de fecha.

    Retorna True si la columna es válida, False en caso contrario.
    """
    if columna_fecha not in dataframe.columns:
        logger.warning("La columna de fecha '%s' no existe en el DataFrame", columna_fecha)
        return False

    total_filas = len(dataframe)
    if total_filas == 0:
        logger.warning("El DataFrame está vacío, no se puede validar la columna de fecha")
        return False

    fechas_parseadas = pd.to_datetime(dataframe[columna_fecha], errors="coerce")
    proporcion_nat = fechas_parseadas.isna().sum() / total_filas

    if proporcion_nat > UMBRAL_NAT:
        logger.warning(
            "La columna '%s' tiene %.1f%% de fechas inválidas (umbral: %.0f%%)",
            columna_fecha,
            proporcion_nat * 100,
            UMBRAL_NAT * 100,
        )
        return False

    return True


def validate_no_future_leakage(dataframe: pd.DataFrame) -> bool:
    """
    Verifica que ninguna columna del DataFrame use datos futuros relativos a la
    fecha del registro. Heurístico: detecta columnas cuyos nombres contienen
    las palabras clave 'shift', 'next', 'futuro', 'siguiente' o 'proximo'.

    Parámetros:
        dataframe: DataFrame a validar.

    Retorna True si no se detecta leakage, False en caso contrario.
    """
    columnas_sospechosas = [
        columna
        for columna in dataframe.columns
        if any(patron in columna.lower() for patron in PATRONES_LEAKAGE)
    ]
    if columnas_sospechosas:
        logger.warning(
            "Posible data leakage del futuro detectado en columnas: %s", columnas_sospechosas
        )
        return False
    return True
