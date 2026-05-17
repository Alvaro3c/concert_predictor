from app.libraries.catalog_utils import extraer_valores_unicos_de_campo

RUTA_CONCIERTOS_ENRIQUECIDOS = "data/processed/conciertos_enriquecido.jsonl"


def obtener_paises_disponibles() -> list[str]:
    """Devuelve la lista ordenada de países únicos presentes en el dataset enriquecido."""
    return extraer_valores_unicos_de_campo(RUTA_CONCIERTOS_ENRIQUECIDOS, "pais")


def obtener_artistas_disponibles() -> list[str]:
    """Devuelve la lista ordenada de artistas únicos presentes en el dataset enriquecido."""
    return extraer_valores_unicos_de_campo(RUTA_CONCIERTOS_ENRIQUECIDOS, "artista")
