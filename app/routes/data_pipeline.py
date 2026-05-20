from fastapi import APIRouter, Depends, HTTPException, Query
from app.libraries.admin_auth import verificar_clave_admin
from app.libraries.hf_dataset_utils import subir_dataset_a_hf, RUTA_LOCAL_DATASET
from app.modules.scrapper import scrapear_conciertos
from app.modules.normalisation import normalise_concerts
from app.modules.features.artist_global_features import enriquecer_conciertos
from app.modules.features.artist_tour_features import enriquecer_tour_features
from app.modules.features.artist_country_features import enriquecer_country_features
from app.modules.features.country_global_features import enriquecer_country_global_features
from app.modules.features.context_features import enriquecer_context_features
from app.modules.train_model import train_pipeline

router = APIRouter(prefix="/data-pipeline", tags=["data_pipeline"], dependencies=[Depends(verificar_clave_admin)])

RUTA_CONCIERTOS_ENRIQUECIDOS = "data/processed/conciertos_enriquecido.jsonl"

RUTA_CONCERTS_RAW = "data/raw/concerts.jsonl"

@router.post("/scrape")
def scrape_concerts(artists_names: list[str] = Query(...), until_year: int | None = Query(None)):
    """Scrapea conciertos de concertarchives para una lista de artistas y los guarda en JSONL."""
    try:
        return scrapear_conciertos(artists_names, until_year)
    except Exception:
        raise HTTPException(status_code=500, detail="Error durante el scraping.")


@router.post("/process/normalise")
def process_normalise():
    """Normaliza el campo fecha del JSONL crudo y guarda el resultado en interim."""
    try:
        return normalise_concerts()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Fichero de datos raw no encontrado.")
    except Exception:
        raise HTTPException(status_code=500, detail="Error durante la normalización.")


@router.post("/process/enrich-features")
def process_enrich_features():
    """Pipeline completo de features: global → tour → country → country_global → context. Sube el resultado a HuggingFace."""
    try:
        resultado_global         = enriquecer_conciertos()
        resultado_tour           = enriquecer_tour_features()
        resultado_country        = enriquecer_country_features()
        resultado_country_global = enriquecer_country_global_features()
        resultado_context        = enriquecer_context_features()
        resultado_subida         = subir_dataset_a_hf(RUTA_CONCIERTOS_ENRIQUECIDOS)
        return {
            "global":         resultado_global,
            "tour":           resultado_tour,
            "country":        resultado_country,
            "country_global": resultado_country_global,
            "context":        resultado_context,
            "subida_hf":      resultado_subida,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Fichero de datos procesados no encontrado.")
    except Exception:
        raise HTTPException(status_code=500, detail="Error durante el enriquecimiento de features.")


@router.post("/upload-dataset")
def upload_dataset():
    """Sube el dataset enriquecido actual a HuggingFace sin necesidad de regenerarlo."""
    try:
        return subir_dataset_a_hf(RUTA_LOCAL_DATASET)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset local no encontrado.")
    except Exception:
        raise HTTPException(status_code=500, detail="Error al subir el dataset.")


@router.post("/train/model-predict")
def train_model():
    """Entrena/reentrena el modelo con los datos procesados."""
    try:
        return train_pipeline(RUTA_CONCIERTOS_ENRIQUECIDOS, "models/model_predict")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset de entrenamiento no encontrado.")
    except ValueError:
        raise HTTPException(status_code=422, detail="Los datos no superaron la validación previa al entrenamiento.")
    except Exception:
        raise HTTPException(status_code=500, detail="Error durante el entrenamiento del modelo.")
