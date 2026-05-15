import pandas as pd
from datetime import datetime, timedelta


def conciertos_en_ventana(fechas: list[datetime], fecha_actual: datetime, dias: int) -> int:
    """
    Cuenta cuántas fechas caen dentro de la ventana [fecha_actual - dias, fecha_actual).

    No incluye fecha_actual (sin data leakage). La lista de fechas debe contener
    solo fechas ya validadas como estrictamente anteriores a fecha_actual.

    Parámetros:
        fechas: lista de objetos datetime a evaluar.
        fecha_actual: fecha de referencia del registro en curso.
        dias: amplitud de la ventana en días (ej. 365, 1095, 1825).

    Devuelve el número de fechas dentro de la ventana como entero.
    """
    if not fechas:
        return 0
    limite_inferior = fecha_actual - timedelta(days=dias)
    serie = pd.Series(fechas)
    return int((serie >= limite_inferior).sum())


def dias_desde_ultimo_concierto(fechas: list[datetime], fecha_actual: datetime) -> int | None:
    """
    Calcula los días transcurridos desde la fecha más reciente en `fechas`.

    Todas las fechas de la lista deben ser estrictamente anteriores a fecha_actual.
    Devuelve None si la lista está vacía.
    """
    if not fechas:
        return None
    serie = pd.Series(fechas)
    fecha_mas_reciente = serie.max()
    return int((fecha_actual - fecha_mas_reciente).days)


def calcular_tendencia_actividad(fechas: list[datetime], fecha_actual: datetime) -> float | None:
    """
    Calcula el ratio entre los conciertos del último año y la media anual de los 5 años previos.

    Fórmula: conciertos_ultimo_año / (total_conciertos_5_años / 5).
    Devuelve None si no hay conciertos en la ventana de 5 años (datos insuficientes).

    Todas las fechas de la lista deben ser estrictamente anteriores a fecha_actual.
    """
    if not fechas:
        return None
    serie = pd.Series(fechas)
    limite_ultimo_año = fecha_actual - timedelta(days=365)
    limite_5_años = fecha_actual - timedelta(days=1825)

    conciertos_ultimo_año = int((serie >= limite_ultimo_año).sum())
    conciertos_5_años = int((serie >= limite_5_años).sum())

    if conciertos_5_años == 0:
        return None

    media_anual_5_años = conciertos_5_años / 5.0
    return round(conciertos_ultimo_año / media_anual_5_años, 4)


def paises_distintos_en_ventana(
    fechas_paises: list[tuple[datetime, str]],
    fecha_actual: datetime,
    dias: int,
) -> int:
    """
    Cuenta el número de países distintos visitados dentro de [fecha_actual - dias, fecha_actual).

    Parámetros:
        fechas_paises: lista de tuplas (fecha, pais) ya validadas como estrictamente
                       anteriores a fecha_actual.
        fecha_actual: fecha de referencia del registro en curso.
        dias: amplitud de la ventana en días.

    Devuelve el número de países únicos como entero. Devuelve 0 si la lista está vacía.
    """
    if not fechas_paises:
        return 0
    limite_inferior = fecha_actual - timedelta(days=dias)
    paises_en_ventana = {
        pais
        for fecha, pais in fechas_paises
        if fecha >= limite_inferior and pais
    }
    return len(paises_en_ventana)


def calcular_ratio_expansion_artista(
    fechas: list[datetime],
    fecha_actual: datetime,
) -> float | None:
    """
    Calcula el ratio de expansión/contracción de la carrera de un artista.

    Compara la tasa de conciertos por año en los últimos 2 años con la tasa
    del período anterior (de hace 2 a 5 años).

    Fórmula: (conciertos_ultimos_2_años / 2) / (conciertos_de_2_a_5_años / 3)

    Resultado:
        > 1.0 → expansión (el artista está más activo ahora que antes)
        ≈ 1.0 → estable
        < 1.0 → contracción o hiato (el artista está menos activo que antes)

    Devuelve None si no hay conciertos en la ventana 2-5 años (datos insuficientes).

    Todas las fechas de la lista deben ser estrictamente anteriores a fecha_actual.
    """
    if not fechas:
        return None

    serie = pd.Series(fechas)
    limite_2_años = fecha_actual - timedelta(days=730)
    limite_5_años = fecha_actual - timedelta(days=1825)

    conciertos_recientes = int((serie >= limite_2_años).sum())
    conciertos_anteriores = int(((serie >= limite_5_años) & (serie < limite_2_años)).sum())

    tasa_reciente = conciertos_recientes / 2.0
    tasa_anterior = conciertos_anteriores / 3.0

    if tasa_anterior == 0:
        return None

    return round(tasa_reciente / tasa_anterior, 4)


def años_observados_artista(fechas: list[datetime], fecha_actual: datetime) -> float | None:
    """
    Calcula los años transcurridos entre el primer concierto conocido y fecha_actual.

    Todas las fechas de la lista deben ser estrictamente anteriores a fecha_actual.
    Devuelve None si la lista está vacía.
    """
    if not fechas:
        return None
    serie = pd.Series(fechas)
    primera_fecha = serie.min()
    return round((fecha_actual - primera_fecha).days / 365.25, 4)
