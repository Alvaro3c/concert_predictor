import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.libraries.date_utils import parsear_fecha
from app.libraries.window_utils import (
    conciertos_en_ventana,
    dias_desde_ultimo_concierto,
    paises_distintos_en_ventana,
)

RUTA_ENRIQUECIDO = "data/processed/conciertos_enriquecido.jsonl"

logger = logging.getLogger(__name__)


def calcular_features_artista_tour(
    registros_previos_artista: list[dict],
    fecha_actual: datetime,
) -> dict:
    """
    Calcula las features de actividad en gira de un artista para un registro dado.

    Recibe únicamente los registros del mismo artista con fecha estrictamente anterior
    a fecha_actual, garantizando la ausencia de data leakage.

    Parámetros:
        registros_previos_artista: lista de dicts con al menos los campos "fecha"
                                   (str YYYY-MM-DD) y "pais" (str).
        fecha_actual: fecha del registro en curso como objeto datetime.

    Devuelve un dict con las keys:
        conciertos_artista_ultimos_30_dias, conciertos_artista_ultimos_90_dias,
        en_gira_activa, dias_desde_concierto_anterior, paises_distintos_ultimos_30_dias.
    Cualquier feature que no pueda calcularse por falta de datos se devuelve como None.
    """
    fechas_previas = [
        parsear_fecha(registro.get("fecha"))
        for registro in registros_previos_artista
    ]
    fechas_previas = [fecha for fecha in fechas_previas if fecha is not None]

    fechas_paises_previas = [
        (parsear_fecha(registro.get("fecha")), registro.get("pais", ""))
        for registro in registros_previos_artista
    ]
    fechas_paises_previas = [
        (fecha, pais)
        for fecha, pais in fechas_paises_previas
        if fecha is not None
    ]

    conciertos_30_dias = conciertos_en_ventana(fechas_previas, fecha_actual, 30)

    return {
        "conciertos_artista_ultimos_30_dias": conciertos_30_dias,
        "conciertos_artista_ultimos_90_dias": conciertos_en_ventana(fechas_previas, fecha_actual, 90),
        "en_gira_activa": conciertos_30_dias >= 3,
        "dias_desde_concierto_anterior": dias_desde_ultimo_concierto(fechas_previas, fecha_actual),
        "paises_distintos_ultimos_30_dias": paises_distintos_en_ventana(fechas_paises_previas, fecha_actual, 30),
    }


def _convertir_nan_a_none(registro: dict) -> dict:
    """Reemplaza valores NaN de pandas/numpy por None para serialización JSON correcta."""
    return {
        clave: (None if isinstance(valor, float) and pd.isna(valor) else valor)
        for clave, valor in registro.items()
    }


def enriquecer_tour_features() -> dict:
    """
    Pipeline de enriquecimiento con artist tour features.

    Lee data/processed/conciertos_enriquecido.jsonl (ya enriquecido con global features),
    añade las tour features y sobreescribe el mismo fichero.

    1. Carga el JSONL en un DataFrame.
    2. Agrupa por artista para evitar iterar el dataset completo por cada registro.
    3. Para cada registro, calcula las tour features usando solo los conciertos del mismo
       artista con fecha estrictamente anterior (sin data leakage).
    4. Sobreescribe data/processed/conciertos_enriquecido.jsonl con las nuevas columnas.

    Devuelve un dict resumen con el total de registros procesados y la ruta de destino.
    """
    ruta_enriquecido = Path(RUTA_ENRIQUECIDO)
    if not ruta_enriquecido.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {RUTA_ENRIQUECIDO}")

    dataframe = pd.read_json(ruta_enriquecido, lines=True, dtype=False)
    total_cargados = len(dataframe)
    logger.info("Cargados %d registros desde %s", total_cargados, RUTA_ENRIQUECIDO)

    dataframe["fecha_dt"] = pd.to_datetime(dataframe["fecha"], format="%Y-%m-%d", errors="coerce")
    dataframe = dataframe.sort_values(["artista", "fecha_dt"]).reset_index(drop=True)

    columnas_originales = [columna for columna in dataframe.columns if columna != "fecha_dt"]

    resultados = []
    total_sin_fecha_valida = 0
    total_artistas = dataframe["artista"].nunique()
    artistas_procesados = 0
    print(f"[2/5 Tour] Iniciando — {total_cargados} registros, {total_artistas} artistas", flush=True)

    for nombre_artista, grupo_artista in dataframe.groupby("artista", sort=False):
        grupo_artista = grupo_artista.reset_index(drop=True)
        registros_artista = grupo_artista.to_dict(orient="records")
        artistas_procesados += 1
        if artistas_procesados % 5 == 0 or artistas_procesados == total_artistas:
            print(f"[2/5 Tour]  {artistas_procesados}/{total_artistas} artistas — {nombre_artista} ({len(registros_artista)} conciertos) — {len(resultados)} registros acumulados", flush=True)

        for indice, registro_actual in enumerate(registros_artista):
            fecha_dt_actual = registro_actual.get("fecha_dt")

            registro_enriquecido = {
                clave: registro_actual[clave]
                for clave in columnas_originales
                if clave in registro_actual
            }

            if pd.isna(fecha_dt_actual):
                for nombre_feature in (
                    "conciertos_artista_ultimos_30_dias",
                    "conciertos_artista_ultimos_90_dias",
                    "en_gira_activa",
                    "dias_desde_concierto_anterior",
                    "paises_distintos_ultimos_30_dias",
                ):
                    registro_enriquecido[nombre_feature] = None
                resultados.append(registro_enriquecido)
                total_sin_fecha_valida += 1
                continue

            fecha_actual_dt = fecha_dt_actual.to_pydatetime()

            registros_previos = [
                {clave: r[clave] for clave in columnas_originales if clave in r}
                for r in registros_artista[:indice]
                if not pd.isna(r.get("fecha_dt")) and r["fecha_dt"] < fecha_dt_actual
            ]

            features = calcular_features_artista_tour(registros_previos, fecha_actual_dt)
            registro_enriquecido.update(features)
            resultados.append(registro_enriquecido)

    print(f"[2/5 Tour]  Todos los artistas procesados. Escribiendo fichero...", flush=True)
    with open(ruta_enriquecido, "w", encoding="utf-8") as archivo_salida:
        for registro in resultados:
            linea = _convertir_nan_a_none(registro)
            archivo_salida.write(json.dumps(linea, ensure_ascii=False) + "\n")

    print(f"[2/5 Tour] Completado — {len(resultados)} registros escritos en {RUTA_ENRIQUECIDO}", flush=True)
    resumen = {
        "total_procesados": len(resultados),
        "sin_fecha_valida": total_sin_fecha_valida,
        "archivo_destino": RUTA_ENRIQUECIDO,
    }

    logger.info(
        "Tour features completadas — procesados: %d | sin fecha válida: %d",
        len(resultados),
        total_sin_fecha_valida,
    )

    return resumen
