from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.modules.predict_model import predecir_retorno

router = APIRouter(tags=["prediction"])
limitador = Limiter(key_func=get_remote_address)


@router.get("/predict")
@limitador.limit("15/minute")
def predict(request: Request, artist: str, country: str) -> dict:
    try:
        return predecir_retorno(artist, country)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artista o país no encontrado en el dataset.")
    except ValueError:
        raise HTTPException(status_code=422, detail="Datos de entrada inválidos.")
    except RuntimeError:
        raise HTTPException(status_code=502, detail="Error al conectar con el servicio de explicaciones. Inténtalo de nuevo.")
    except Exception:
        raise HTTPException(status_code=500, detail="Error interno del servidor.")
