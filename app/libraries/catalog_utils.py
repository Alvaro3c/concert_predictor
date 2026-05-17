import json
import logging
from pathlib import Path

from app.libraries.hf_dataset_utils import asegurar_dataset_local

logger = logging.getLogger(__name__)


def extraer_valores_unicos_de_campo(ruta_archivo: str, nombre_campo: str) -> list[str]:
    """Lee el JSONL línea a línea y devuelve los valores únicos no nulos de un campo dado, ordenados."""
    asegurar_dataset_local(ruta_archivo)
    ruta = Path(ruta_archivo)
    if not ruta.exists():
        raise FileNotFoundError(f"Archivo de datos no encontrado: {ruta_archivo}")

    valores_unicos: set[str] = set()

    with open(ruta, encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            registro = json.loads(linea)
            valor = registro.get(nombre_campo)
            if valor and isinstance(valor, str):
                valores_unicos.add(valor)

    return sorted(valores_unicos)
