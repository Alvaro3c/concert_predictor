# Configuración global del pipeline de entrenamiento

# Filtrar registros cuyo gap entre concierto y siguiente atraviesa el período COVID (2020-2021).
# Estos gaps son artificialmente largos y no reflejan el comportamiento normal de un artista.
FILTRAR_GAPS_COVID: bool = True

FECHA_INICIO_COVID: str = "2020-01-01"
FECHA_FIN_COVID: str = "2021-12-31"
