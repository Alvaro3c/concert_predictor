from fastapi import APIRouter, HTTPException

from app.modules.catalog import obtener_paises_disponibles, obtener_artistas_disponibles

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/countries")
def get_countries() -> list[str]:
    """Devuelve los países únicos disponibles en el dataset para alimentar el formulario del frontend."""
    try:
        return obtener_paises_disponibles()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/artists")
def get_artists() -> list[str]:
    """Devuelve los artistas únicos disponibles en el dataset para alimentar el formulario del frontend."""
    try:
        return obtener_artistas_disponibles()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
