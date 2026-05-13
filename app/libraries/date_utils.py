from datetime import datetime

FORMATO_FECHA = "%Y-%m-%d"


def parsear_fecha(fecha_str: str | None) -> datetime | None:
    """
    Parsea un string 'YYYY-MM-DD' a un objeto datetime.

    Devuelve None si el valor es None, vacío o no tiene formato válido.
    """
    if fecha_str is None or not isinstance(fecha_str, str) or not fecha_str.strip():
        return None
    try:
        return datetime.strptime(fecha_str.strip(), FORMATO_FECHA)
    except ValueError:
        return None


def dias_entre_fechas(fecha_anterior: datetime, fecha_actual: datetime) -> int:
    """
    Calcula el número de días entre dos fechas.

    Devuelve un entero positivo si fecha_actual es posterior a fecha_anterior.
    """
    return (fecha_actual - fecha_anterior).days


def años_entre_fechas(fecha_primera: datetime, fecha_actual: datetime) -> float:
    """
    Calcula el número de años decimales entre dos fechas usando 365.25 días/año.

    Devuelve un float positivo si fecha_actual es posterior a fecha_primera.
    """
    return (fecha_actual - fecha_primera).days / 365.25


_MES_A_ESTACION: dict[int, str] = {
    1: "invierno", 2: "invierno",
    3: "primavera", 4: "primavera", 5: "primavera",
    6: "verano",    7: "verano",    8: "verano",
    9: "otoño",    10: "otoño",   11: "otoño",
    12: "invierno",
}

_INICIO_COVID = datetime(2020, 3, 1)
_FIN_COVID = datetime(2021, 12, 31)
_POST_COVID = datetime(2022, 1, 1)


def obtener_estacion(fecha: datetime) -> str:
    """
    Devuelve la estación del año correspondiente a la fecha (hemisferio norte).

    Criterio por mes:
        invierno → diciembre, enero, febrero    (meses 12, 1, 2)
        primavera → marzo, abril, mayo           (meses 3, 4, 5)
        verano    → junio, julio, agosto          (meses 6, 7, 8)
        otoño     → septiembre, octubre, noviembre (meses 9, 10, 11)

    Parámetros:
        fecha: objeto datetime con la fecha del registro.

    Devuelve uno de: "invierno", "primavera", "verano", "otoño".
    """
    return _MES_A_ESTACION[fecha.month]


def calcular_es_post_covid(fecha: datetime) -> bool:
    """
    Indica si la fecha es posterior al periodo COVID (estrictamente después del 2022-01-01).

    Parámetros:
        fecha: objeto datetime con la fecha del registro.

    Devuelve True si fecha > 2022-01-01, False en caso contrario.
    """
    return fecha > _POST_COVID


def calcular_es_periodo_covid(fecha: datetime) -> bool:
    """
    Indica si la fecha cae dentro del periodo de restricciones COVID.

    Periodo considerado: 2020-03-01 ≤ fecha ≤ 2021-12-31 (inclusive en ambos extremos).

    Parámetros:
        fecha: objeto datetime con la fecha del registro.

    Devuelve True si la fecha está dentro del periodo, False en caso contrario.
    """
    return _INICIO_COVID <= fecha <= _FIN_COVID
