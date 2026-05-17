from fastapi import APIRouter, HTTPException

from app.modules.predict_model import predecir_retorno

router = APIRouter(tags=["prediction"])


@router.get("/predict")
def predict(artist: str, country: str) -> dict:
    try:
        return predecir_retorno(artist, country)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
