import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.libraries.date_utils import parsear_fecha
from app.libraries.gap_utils import (
    calcular_gaps,
    gap_medio,
    gap_mediano,
    gap_std,
    gap_minimo,
    gap_maximo,
    ultimo_gap,
)
from app.libraries.window_utils import dias_desde_ultimo_concierto

RUTA_ENRIQUECIDO = "data/processed/conciertos_enriquecido.jsonl"

logger = logging.getLogger(__name__)


def calcular_features_artista_pais(
    registros_previos_artista: list[dict],
    pais: str,
    fecha_actual: datetime,
) -> dict:
    """
    Calcula las features de relación artista × país para un registro dado.

    Recibe todos los registros del mismo artista con fecha estrictamente anterior
    a fecha_actual. Filtra internamente por país para las features específicas del país
    y usa el total de registros para proporciones y ranking.

    Parámetros:
        registros_previos_artista: lista de dicts con al menos los campos "fecha"
                                   (str YYYY-MM-DD) y "pais" (str).
        pais: país del registro en curso.
        fecha_actual: fecha del registro en curso como objeto datetime.

    Devuelve un dict con las keys:
        visitas_previas_al_pais, es_primera_visita_observada,
        dias_desde_ultima_visita_pais, gap_medio_pais, gap_mediano_pais,
        gap_std_pais, gap_min_pais, gap_max_pais, ultimo_gap_pais,
        tendencia_gap_pais, proporcion_visitas_pais, rank_pais_para_artista.
    Cualquier feature que no pueda calcularse por falta de datos se devuelve como None.
    """
    total_previos = len(registros_previos_artista)

    # Fechas del artista en el país concreto, ordenadas cronológicamente
    fechas_previas_pais = sorted([
        fecha
        for registro in registros_previos_artista
        if registro.get("pais") == pais
        for fecha in [parsear_fecha(registro.get("fecha"))]
        if fecha is not None
    ])

    visitas_previas = len(fechas_previas_pais)

    gaps = calcular_gaps(fechas_previas_pais)
    gap_medio_valor = gap_medio(gaps)
    ultimo_gap_valor = ultimo_gap(gaps)

    if ultimo_gap_valor is not None and gap_medio_valor is not None and gap_medio_valor > 0:
        tendencia_gap = round(ultimo_gap_valor / gap_medio_valor, 4)
    else:
        tendencia_gap = None

    # Ranking del país: posición 1-based usando dense rank por visitas descendentes
    if total_previos > 0 and visitas_previas > 0:
        conteos_pais = Counter(
            registro.get("pais", "")
            for registro in registros_previos_artista
            if registro.get("pais")
        )
        rank_pais = len({conteo for pais_key, conteo in conteos_pais.items() if conteo > visitas_previas}) + 1
    else:
        rank_pais = None

    return {
        "visitas_previas_al_pais": visitas_previas,
        "es_primera_visita_observada": visitas_previas == 0,
        "dias_desde_ultima_visita_pais": dias_desde_ultimo_concierto(fechas_previas_pais, fecha_actual),
        "gap_medio_pais": gap_medio_valor,
        "gap_mediano_pais": gap_mediano(gaps),
        "gap_std_pais": gap_std(gaps),
        "gap_min_pais": gap_minimo(gaps),
        "gap_max_pais": gap_maximo(gaps),
        "ultimo_gap_pais": ultimo_gap_valor,
        "tendencia_gap_pais": tendencia_gap,
        "proporcion_visitas_pais": round(visitas_previas / total_previos, 4) if total_previos > 0 else None,
        "rank_pais_para_artista": rank_pais,
    }


def _convertir_nan_a_none(registro: dict) -> dict:
    """Reemplaza valores NaN de pandas/numpy por None para serialización JSON correcta."""
    return {
        clave: (None if isinstance(valor, float) and pd.isna(valor) else valor)
        for clave, valor in registro.items()
    }


NOMBRES_FEATURES_COUNTRY = (
    "visitas_previas_al_pais",
    "es_primera_visita_observada",
    "dias_desde_ultima_visita_pais",
    "gap_medio_pais",
    "gap_mediano_pais",
    "gap_std_pais",
    "gap_min_pais",
    "gap_max_pais",
    "ultimo_gap_pais",
    "tendencia_gap_pais",
    "proporcion_visitas_pais",
    "rank_pais_para_artista",
)


def enriquecer_country_features() -> dict:
    """
    Pipeline de enriquecimiento con artist × country features.

    Lee data/processed/conciertos_enriquecido.jsonl, añade las country features
    y sobreescribe el mismo fichero.

    1. Carga el JSONL en un DataFrame.
    2. Agrupa por artista; dentro de cada grupo, pre-agrupa por país antes del
       bucle de registros para evitar re-escanear el grupo completo en cada iteración.
    3. Para cada registro, calcula las features usando solo los conciertos del mismo
       artista con fecha estrictamente anterior (sin data leakage).
    4. Sobreescribe data/processed/conciertos_enriquecido.jsonl con las nuevas columnas.

    Devuelve un dict resumen con total de registros procesados y la ruta de destino.
    """
    ruta_enriquecido = Path(RUTA_ENRIQUECIDO)
    if not ruta_enriquecido.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {RUTA_ENRIQUECIDO}")

    dataframe = pd.read_json(ruta_enriquecido, lines=True, dtype=False)
    logger.info("Cargados %d registros desde %s", len(dataframe), RUTA_ENRIQUECIDO)

    dataframe["fecha_dt"] = pd.to_datetime(dataframe["fecha"], format="%Y-%m-%d", errors="coerce")
    dataframe = dataframe.sort_values(["artista", "fecha_dt"]).reset_index(drop=True)

    columnas_originales = [columna for columna in dataframe.columns if columna != "fecha_dt"]

    resultados = []
    total_sin_fecha_valida = 0
    total_artistas = dataframe["artista"].nunique()
    artistas_procesados = 0
    print(f"[3/5 Country] Iniciando — {len(dataframe)} registros, {total_artistas} artistas", flush=True)

    for nombre_artista, grupo_artista in dataframe.groupby("artista", sort=False):
        grupo_artista = grupo_artista.reset_index(drop=True)
        registros_artista = grupo_artista.to_dict(orient="records")
        artistas_procesados += 1
        if artistas_procesados % 5 == 0 or artistas_procesados == total_artistas:
            print(f"[3/5 Country]  {artistas_procesados}/{total_artistas} artistas — {nombre_artista} ({len(registros_artista)} conciertos) — {len(resultados)} registros acumulados", flush=True)

        # Pre-agrupar por país para que el filtro interno de calcular_features_artista_pais
        # opere sobre subconjuntos pequeños en lugar de escanear todo el grupo del artista
        indices_por_pais: dict[str, list[int]] = {}
        for indice_registro, registro in enumerate(registros_artista):
            pais_registro = registro.get("pais", "")
            if pais_registro not in indices_por_pais:
                indices_por_pais[pais_registro] = []
            indices_por_pais[pais_registro].append(indice_registro)

        for indice, registro_actual in enumerate(registros_artista):
            fecha_dt_actual = registro_actual.get("fecha_dt")

            registro_enriquecido = {
                clave: registro_actual[clave]
                for clave in columnas_originales
                if clave in registro_actual
            }

            if pd.isna(fecha_dt_actual):
                for nombre_feature in NOMBRES_FEATURES_COUNTRY:
                    registro_enriquecido[nombre_feature] = None
                resultados.append(registro_enriquecido)
                total_sin_fecha_valida += 1
                continue

            fecha_actual_dt = fecha_dt_actual.to_pydatetime()
            pais_actual = registro_actual.get("pais", "")

            # Registros previos del artista en cualquier país (para proporción y ranking)
            registros_previos_artista = [
                {clave: r[clave] for clave in columnas_originales if clave in r}
                for r in registros_artista[:indice]
                if not pd.isna(r.get("fecha_dt")) and r["fecha_dt"] < fecha_dt_actual
            ]

            features = calcular_features_artista_pais(
                registros_previos_artista, pais_actual, fecha_actual_dt
            )
            registro_enriquecido.update(features)
            resultados.append(registro_enriquecido)

    print(f"[3/5 Country]  Todos los artistas procesados. Escribiendo fichero...", flush=True)
    with open(ruta_enriquecido, "w", encoding="utf-8") as archivo_salida:
        for registro in resultados:
            linea = _convertir_nan_a_none(registro)
            archivo_salida.write(json.dumps(linea, ensure_ascii=False) + "\n")

    print(f"[3/5 Country] Completado — {len(resultados)} registros escritos en {RUTA_ENRIQUECIDO}", flush=True)
    resumen = {
        "total_procesados": len(resultados),
        "sin_fecha_valida": total_sin_fecha_valida,
        "archivo_destino": RUTA_ENRIQUECIDO,
    }

    logger.info(
        "Country features completadas — procesados: %d | sin fecha válida: %d",
        len(resultados),
        total_sin_fecha_valida,
    )

    return resumen
