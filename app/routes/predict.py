from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["prediction"])


@router.get("/predict")
def predict(artist: str, country: str) -> str:
    # TODO: reemplazar con el modelo de lenguaje natural real
    return f"El artista {artist} tocará en {country} próximamente."