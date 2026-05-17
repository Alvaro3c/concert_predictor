from fastapi import APIRouter, HTTPException, Query
from app.modules.scrapper import scrapear_conciertos
from app.modules.normalisation import normalise_concerts
from app.modules.features.artist_global_features import enriquecer_conciertos
from app.modules.features.artist_tour_features import enriquecer_tour_features
from app.modules.features.artist_country_features import enriquecer_country_features
from app.modules.features.country_global_features import enriquecer_country_global_features
from app.modules.features.context_features import enriquecer_context_features
from app.modules.train_model import train_pipeline

router = APIRouter(prefix="/data-pipeline", tags=["data_pipeline"])

RUTA_CONCIERTOS_ENRIQUECIDOS = "data/processed/conciertos_enriquecido.jsonl"

RUTA_CONCERTS_RAW = "data/raw/concerts.jsonl"

@router.post("/scrape")
def scrape_concerts(artists_names: list[str] = Query(...), until_year: int | None = Query(None)):
    """Scrapea conciertos de concertarchives para una lista de artistas y los guarda en JSONL."""
    try:
        return scrapear_conciertos(artists_names, until_year)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/process/normalise")
def process_normalise():
    """Normaliza el campo fecha del JSONL crudo y guarda el resultado en interim."""
    try:
        return normalise_concerts()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/process/enrich-features")
def process_enrich_features():
    """Pipeline completo de features: global → tour → country → country_global → context."""
    try:
        resultado_global         = enriquecer_conciertos()
        resultado_tour           = enriquecer_tour_features()
        resultado_country        = enriquecer_country_features()
        resultado_country_global = enriquecer_country_global_features()
        resultado_context        = enriquecer_context_features()
        return {
            "global":         resultado_global,
            "tour":           resultado_tour,
            "country":        resultado_country,
            "country_global": resultado_country_global,
            "context":        resultado_context,
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/train/model-predict")
def train_model():
    """Entrena/reentrena el modelo con los datos procesados."""
    try:
        return train_pipeline(RUTA_CONCIERTOS_ENRIQUECIDOS, "models/model_predict")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


