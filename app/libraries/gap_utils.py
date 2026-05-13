import pandas as pd
from datetime import datetime


def calcular_gaps(fechas: list[datetime]) -> list[int]:
    """
    Calcula los gaps en días entre visitas consecutivas.

    Parámetros:
        fechas: lista de datetimes ordenada cronológicamente.

    Devuelve una lista de enteros con las diferencias en días entre cada par
    de visitas consecutivas. Devuelve lista vacía si hay menos de 2 fechas.
    """
    if len(fechas) < 2:
        return []
    serie = pd.Series(fechas).sort_values().reset_index(drop=True)
    return [int((serie[i] - serie[i - 1]).days) for i in range(1, len(serie))]


def gap_medio(gaps: list[int]) -> float | None:
    """
    Calcula la media de los gaps en días.

    Devuelve None si la lista está vacía.
    """
    if not gaps:
        return None
    return round(float(pd.Series(gaps).mean()), 4)


def gap_mediano(gaps: list[int]) -> float | None:
    """
    Calcula la mediana de los gaps en días.

    Devuelve None si la lista está vacía.
    """
    if not gaps:
        return None
    return round(float(pd.Series(gaps).median()), 4)


def gap_std(gaps: list[int]) -> float | None:
    """
    Calcula la desviación estándar de los gaps en días (ddof=0, población).

    Devuelve None si la lista está vacía.
    """
    if not gaps:
        return None
    return round(float(pd.Series(gaps).std(ddof=0)), 4)


def gap_minimo(gaps: list[int]) -> int | None:
    """
    Devuelve el gap mínimo registrado en días.

    Devuelve None si la lista está vacía.
    """
    if not gaps:
        return None
    return int(min(gaps))


def gap_maximo(gaps: list[int]) -> int | None:
    """
    Devuelve el gap máximo registrado en días.

    Devuelve None si la lista está vacía.
    """
    if not gaps:
        return None
    return int(max(gaps))


def ultimo_gap(gaps: list[int]) -> int | None:
    """
    Devuelve el gap más reciente, es decir, el último elemento de la lista.

    Devuelve None si la lista está vacía.
    """
    if not gaps:
        return None
    return int(gaps[-1])
