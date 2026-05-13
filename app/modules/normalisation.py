import json
import logging
from pathlib import Path

from app.libraries.normalise_utils import (
    normalizar_fecha,
    calcular_es_festival,
    normalizar_artista,
    eliminar_campo_conciertos_previos,
    es_registro_invalido,
    normalizar_pais,
    deduplicar_por_solapamiento,
)

RUTA_FUENTE = "data/raw/concerts.jsonl"
RUTA_DESTINO = "data/interim/normalized_concerts.jsonl"

logger = logging.getLogger(__name__)


def normalise_concerts() -> dict:
    """Normaliza, filtra y deduplica los registros crudos y escribe el resultado en interim."""
    total_procesados = 0
    total_correctos_sin_cambio = 0
    total_normalizados = 0
    total_errores = 0
    total_eliminados = 0
    total_duplicados = 0

    ruta_destino = Path(RUTA_DESTINO)
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)

    # Primera pasada: filtrar y normalizar, agrupando por artista
    registros_por_artista: dict[str, list[dict]] = {}

    with open(RUTA_FUENTE, encoding="utf-8") as archivo_fuente:
        for linea in archivo_fuente:
            linea = linea.strip()
            if not linea:
                continue

            total_procesados += 1
            registro = json.loads(linea)

            # Descartar cruceros y conciertos televisivos
            if es_registro_invalido(registro):
                total_eliminados += 1
                continue

            # Normalizar campo fecha
            valor_fecha_original = registro.get("fecha")
            fecha_normalizada, error_fecha = normalizar_fecha(valor_fecha_original)

            if error_fecha is not None:
                registro["fecha"] = None
                registro["fecha_parse_error"] = error_fecha
                total_errores += 1
            else:
                registro["fecha"] = fecha_normalizada
                if fecha_normalizada == valor_fecha_original:
                    total_correctos_sin_cambio += 1
                else:
                    total_normalizados += 1

            # Normalizar campo fecha_fin con la misma lógica
            valor_fecha_fin_original = registro.get("fecha_fin")
            fecha_fin_normalizada, error_fecha_fin = normalizar_fecha(valor_fecha_fin_original)

            if error_fecha_fin is not None:
                registro["fecha_fin"] = None
                registro["fecha_fin_parse_error"] = error_fecha_fin
            else:
                registro["fecha_fin"] = fecha_fin_normalizada

            registro["es_festival"] = calcular_es_festival(registro["fecha_fin"])

            # Normalizar campo pais
            pais_normalizado, debe_eliminar_pais = normalizar_pais(registro.get("pais"))
            if debe_eliminar_pais:
                total_eliminados += 1
                continue
            registro["pais"] = pais_normalizado

            # Normalizar campo artista
            valor_artista = registro.get("artista")
            if valor_artista is not None:
                registro["artista"] = normalizar_artista(valor_artista)

            eliminar_campo_conciertos_previos(registro)

            clave_artista = registro.get("artista") or ""
            if clave_artista not in registros_por_artista:
                registros_por_artista[clave_artista] = []
            registros_por_artista[clave_artista].append(registro)

    # Segunda pasada: deduplicar por solapamiento de fechas y escribir
    with open(ruta_destino, "w", encoding="utf-8") as archivo_destino:
        for registros in registros_por_artista.values():
            registros_validos, duplicados = deduplicar_por_solapamiento(registros)
            total_duplicados += duplicados
            for registro in registros_validos:
                archivo_destino.write(json.dumps(registro, ensure_ascii=False) + "\n")

    resumen = {
        "total_procesados": total_procesados,
        "correctos_sin_cambio": total_correctos_sin_cambio,
        "normalizados": total_normalizados,
        "errores": total_errores,
        "eliminados": total_eliminados,
        "duplicados": total_duplicados,
    }

    logger.info(
        "Normalización completada — total: %d | correctos: %d | normalizados: %d | errores: %d | eliminados: %d | duplicados: %d",
        total_procesados,
        total_correctos_sin_cambio,
        total_normalizados,
        total_errores,
        total_eliminados,
        total_duplicados,
    )

    return resumen
