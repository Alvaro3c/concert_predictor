"""Utilidades para sincronizar el dataset enriquecido con HuggingFace Hub."""

import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

NOMBRE_FICHERO_DATASET = "conciertos_enriquecido.jsonl"
RUTA_LOCAL_DATASET = "data/processed/conciertos_enriquecido.jsonl"

logger = logging.getLogger(__name__)


def _obtener_repo_id() -> str:
    repo = os.getenv("HF_DATASET_REPO")
    if not repo:
        raise ValueError("HF_DATASET_REPO no está configurado en el entorno")
    return repo


def _obtener_token() -> str:
    token = os.getenv("HF_DATASET_TOKEN")
    if not token:
        raise ValueError("HF_DATASET_TOKEN no está configurado en el entorno")
    return token


def subir_dataset_a_hf(ruta_local: str = RUTA_LOCAL_DATASET) -> dict:
    """Sube el JSONL enriquecido al repositorio de HuggingFace Hub."""
    ruta = Path(ruta_local)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el fichero local para subir: {ruta_local}")

    repo_id = _obtener_repo_id()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(ruta),
        path_in_repo=NOMBRE_FICHERO_DATASET,
        repo_id=repo_id,
        repo_type="dataset",
        token=_obtener_token(),
    )
    logger.info("Dataset subido a HuggingFace — repo: %s", repo_id)
    return {"subido": True, "repo": repo_id, "fichero": NOMBRE_FICHERO_DATASET}


def asegurar_dataset_local(ruta_local: str = RUTA_LOCAL_DATASET) -> str:
    """
    Garantiza que el JSONL esté disponible en disco.
    Si no existe localmente, lo descarga del repositorio de HuggingFace Hub.
    Retorna la ruta local al fichero.
    """
    ruta = Path(ruta_local)
    if ruta.exists():
        return ruta_local

    logger.info("Dataset no encontrado localmente — descargando desde HuggingFace...")
    ruta.parent.mkdir(parents=True, exist_ok=True)

    ruta_cache = hf_hub_download(
        repo_id=_obtener_repo_id(),
        filename=NOMBRE_FICHERO_DATASET,
        repo_type="dataset",
        token=_obtener_token(),
    )
    shutil.copy2(ruta_cache, ruta)
    logger.info("Dataset listo en: %s", ruta_local)
    return ruta_local
