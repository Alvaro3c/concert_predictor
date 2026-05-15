# Configuración global del pipeline de entrenamiento

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
OPTIMIZAR_HIPERPARAMETROS: bool = True
OPTUNA_TRIALS: int = 50
