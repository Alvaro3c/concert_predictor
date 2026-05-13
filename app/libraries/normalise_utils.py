import re
from datetime import date, datetime

PATRON_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

CIUDADES_INVALIDAS = {"OCEAN", "N/A", "Vacío", ""}

NORMALIZACION_PAISES = {
    "NZ": "New Zealand",
    "UK": "United Kingdom",
    "USA": "United States",
    "US": "United States",
    "Vacío": None,
    "N/A": None,
    "": None,
}

VENUES_TV = {
    "Late Show With David Letterman",
    "Last Call With Carson Daly",
    "MTV Studios",
    "TV Total",
    "Jimmy Kimmel Live!",
    "SiriusXM Studios",
    "iHeartRadio Theater",
    "Sunday Brunch",
    "Palladium on Carnival Destiny",
    "Lido Deck on Carnival Destiny",
    "The Ellen DeGeneres Show",
    "HD Radio Sound Space at KROQ",
}

# Captura "Mon DD, YYYY" seguido de sufijo opcional con espacio opcional
PATRON_FECHA_TEXTO = re.compile(
    r"^([A-Za-z]+ \d{1,2}, \d{4})\s*(Buy Tickets|Cancelled|Rescheduled)?\s*$"
)

# Intentamos primero nombre completo (January) y luego abreviado (Jan)
FORMATOS_FECHA = ["%B %d, %Y", "%b %d, %Y"]


def es_formato_iso(valor: str) -> bool:
    return bool(PATRON_ISO.match(valor))


def extraer_parte_fecha(valor: str) -> str | None:
    """Extrae solo 'Mon DD, YYYY' descartando el sufijo Buy Tickets / Cancelled / Rescheduled."""
    coincidencia = PATRON_FECHA_TEXTO.match(valor)
    if coincidencia:
        return coincidencia.group(1)
    return None


def texto_a_fecha_iso(texto_fecha: str) -> str | None:
    """Convierte 'January 15, 2001' o 'Jan 15, 2001' a '2001-01-15'."""
    for formato in FORMATOS_FECHA:
        try:
            fecha = datetime.strptime(texto_fecha.strip(), formato)
            return fecha.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalizar_pais(valor: str | None) -> tuple[str | None, bool]:
    """
    Normaliza el campo pais de un registro.

    Devuelve (pais_normalizado, debe_eliminar):
      - Valor no en diccionario → (valor, False)    sin cambio
      - Valor mapeado a string  → (string, False)   normalizado
      - Valor mapeado a None    → (None, True)       registro a eliminar
      - Valor None              → (None, True)       registro a eliminar
    """
    if valor is None:
        return None, True

    if valor not in NORMALIZACION_PAISES:
        return valor, False

    pais_normalizado = NORMALIZACION_PAISES[valor]
    if pais_normalizado is None:
        return None, True

    return pais_normalizado, False


def deduplicar_por_solapamiento(registros: list[dict]) -> tuple[list[dict], int]:
    """
    Dado una lista de registros del mismo artista, elimina los cuya fecha
    cae dentro del rango [fecha, fecha_fin] de un registro anterior.

    Devuelve (registros_validos, total_duplicados).
    Los registros sin fecha no se pueden comparar y se conservan al final.
    """
    registros_con_fecha = [registro for registro in registros if registro.get("fecha") is not None]
    registros_sin_fecha = [registro for registro in registros if registro.get("fecha") is None]

    registros_ordenados = sorted(registros_con_fecha, key=lambda registro: registro["fecha"])
    registros_validos = []
    total_duplicados = 0
    fecha_fin_activa: date | None = None

    for registro in registros_ordenados:
        fecha_actual = date.fromisoformat(registro["fecha"])

        if fecha_fin_activa is not None and fecha_actual <= fecha_fin_activa:
            total_duplicados += 1
            continue

        registros_validos.append(registro)

        fecha_fin_str = registro.get("fecha_fin")
        if fecha_fin_str is not None:
            fecha_fin_nueva = date.fromisoformat(fecha_fin_str)
            if fecha_fin_activa is None or fecha_fin_nueva > fecha_fin_activa:
                fecha_fin_activa = fecha_fin_nueva

    return registros_validos + registros_sin_fecha, total_duplicados


def es_registro_invalido(registro: dict) -> bool:
    """Devuelve True si el registro corresponde a un crucero o concierto televisivo."""
    ciudad = registro.get("ciudad") or ""
    venue = registro.get("venue") or ""
    return ciudad in CIUDADES_INVALIDAS or venue in VENUES_TV


def eliminar_campo_conciertos_previos(registro: dict) -> dict:
    registro.pop("conciertos_previos_hasta_la_fecha", None)
    return registro


NORMALIZACION_ARTISTAS = {
    "Linking Park": "Linkin Park",
    "Stick To Your Guns 3": "Stick To Your Guns",
    "Twentyone Pilots": "Twenty One Pilots",
}


def normalizar_artista(valor: str) -> str:
    """Convierte 'red-hot-chili-peppers' a 'Red Hot Chili Peppers' y corrige nombres mal escritos."""
    nombre = valor.replace("-", " ").title()
    return NORMALIZACION_ARTISTAS.get(nombre, nombre)


def calcular_es_festival(fecha_fin) -> bool:
    return fecha_fin is not None


def normalizar_fecha(valor) -> tuple[str | None, str | None]:
    """
    Normaliza el campo fecha de un registro.

    Devuelve (fecha_normalizada, error_original):
      - Valor None        → (None, None)           sin cambio
      - Ya es ISO         → (valor, None)           sin cambio
      - Texto con sufijo  → (fecha_iso, None)       normalizado
      - Cualquier otro    → (None, valor_original)  error
    """
    if valor is None:
        return None, None

    valor_str = str(valor)

    if es_formato_iso(valor_str):
        return valor_str, None

    texto_fecha = extraer_parte_fecha(valor_str)
    if texto_fecha is not None:
        fecha_iso = texto_a_fecha_iso(texto_fecha)
        if fecha_iso is not None:
            return fecha_iso, None

    return None, valor_str
