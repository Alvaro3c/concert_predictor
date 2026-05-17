# Configuración global del pipeline de entrenamiento

# Modo ultra-stealth para el scraper. Aumenta los delays base a 4-20 s e introduce
# pausas largas aleatorias de 8-12 min para imitar a un humano que se aleja del teclado.
EXTRA_STEALTH_MODE: bool = False

STEALTH_DELAY_MIN_S: float = 4.0
STEALTH_DELAY_MAX_S: float = 20.0

# Probabilidad por petición de disparar una pausa larga (0.0 = nunca, 1.0 = siempre).
# Con 0.15 ocurre en media 1 de cada 7 páginas de forma impredecible.
STEALTH_PROBABILIDAD_PAUSA_FANTASMA: float = 0.15

STEALTH_PAUSA_LARGA_MIN_S: int = 8 * 60
STEALTH_PAUSA_LARGA_MAX_S: int = 12 * 60

# Filtrar registros cuyo gap entre concierto y siguiente atraviesa el período COVID (2020-2021).
# Estos gaps son artificialmente largos y no reflejan el comportamiento normal de un artista.
FILTRAR_GAPS_COVID: bool = True

FECHA_INICIO_COVID: str = "2020-01-01"
FECHA_FIN_COVID: str = "2021-12-31"

# Umbrales de decisión para la clasificación por tramos.
# Si la probabilidad de B o C supera su umbral, se prefiere ese tramo sobre A.
# Bajar el umbral aumenta el recall de B/C a costa de reducir el de A.
UMBRAL_TRAMO_B: float = 0.25
UMBRAL_TRAMO_C: float = 0.20

# Búsqueda automática de hiperparámetros con Optuna antes de entrenar.
# Activar alarga el entrenamiento varios minutos pero puede mejorar los resultados.
OPTIMIZAR_HIPERPARAMETROS: bool = False
OPTUNA_TRIALS: int = 50
