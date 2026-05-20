from fastapi import APIRouter, HTTPException

from app.modules.catalog import obtener_paises_disponibles, obtener_artistas_disponibles

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/countries")
def get_countries() -> list[str]:
    """Devuelve los países únicos disponibles en el dataset para alimentar el formulario del frontend."""
    try:
        return obtener_paises_disponibles()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset no disponible.")
    except Exception:
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


@router.get("/artists")
def get_artists() -> list[str]:
    """Devuelve los artistas únicos disponibles en el dataset para alimentar el formulario del frontend."""
    try:
        return obtener_artistas_disponibles()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset no disponible.")
    except Exception:
        raise HTTPException(status_code=500, detail="Error interno del servidor.")
