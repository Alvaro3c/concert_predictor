import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.libraries.date_utils import parsear_fecha
from app.libraries.window_utils import conciertos_en_ventana, paises_distintos_en_ventana

RUTA_ENRIQUECIDO = "data/processed/conciertos_enriquecido.jsonl"

logger = logging.getLogger(__name__)


def calcular_features_country_global(
    registros_previos_artista: list[dict],
    pais: str,
    fecha_actual: datetime,
) -> dict:
    """
    Calcula las features globales de país centradas en el artista para un registro dado.

    Solo incluye la perspectiva artista (diversidad geográfica, ratio países/conciertos).
    Las features cross-artista (conciertos_totales_pais_previos, gap_medio_global_pais)
    se pre-computan vectorialmente en el pipeline y se añaden fuera de esta función.

    Parámetros:
        registros_previos_artista: lista de dicts del mismo artista con fecha
                                   estrictamente anterior a fecha_actual.
                                   Campos mínimos: "fecha" (str), "pais" (str),
                                   "ciudad" (str).
        pais: país del registro en curso.
        fecha_actual: fecha del registro en curso como objeto datetime.

    Devuelve un dict con las keys:
        paises_unicos_visitados_total, paises_unicos_visitados_ultimos_3_años,
        paises_unicos_visitados_ultimos_5_años, ciudades_unicas_ultimos_5_años,
        ratio_paises_conciertos_5y.
    Cualquier feature que no pueda calcularse por falta de datos se devuelve como None.
    """
    fechas_paises_artista = [
        (fecha, registro.get("pais", ""))
        for registro in registros_previos_artista
        for fecha in [parsear_fecha(registro.get("fecha"))]
        if fecha is not None and registro.get("pais")
    ]

    fechas_ciudades_artista = [
        (fecha, registro.get("ciudad", ""))
        for registro in registros_previos_artista
        for fecha in [parsear_fecha(registro.get("fecha"))]
        if fecha is not None and registro.get("ciudad")
    ]

    paises_unicos_total = len({pais_r for _, pais_r in fechas_paises_artista})

    paises_5y = paises_distintos_en_ventana(fechas_paises_artista, fecha_actual, 1825)
    conciertos_5y = conciertos_en_ventana(
        [fecha for fecha, _ in fechas_paises_artista], fecha_actual, 1825
    )

    if conciertos_5y > 0:
        ratio_paises_conciertos = round(paises_5y / conciertos_5y, 4)
    else:
        ratio_paises_conciertos = None

    return {
        "paises_unicos_visitados_total": paises_unicos_total,
        "paises_unicos_visitados_ultimos_3_años": paises_distintos_en_ventana(
            fechas_paises_artista, fecha_actual, 1095
        ),
        "paises_unicos_visitados_ultimos_5_años": paises_5y,
        "ciudades_unicas_ultimos_5_años": paises_distintos_en_ventana(
            fechas_ciudades_artista, fecha_actual, 1825
        ),
        "ratio_paises_conciertos_5y": ratio_paises_conciertos,
    }


def _convertir_nan_a_none(registro: dict) -> dict:
    """Reemplaza valores NaN de pandas/numpy por None para serialización JSON correcta."""
    return {
        clave: (None if isinstance(valor, float) and pd.isna(valor) else valor)
        for clave, valor in registro.items()
    }


NOMBRES_FEATURES_COUNTRY_GLOBAL = (
    "paises_unicos_visitados_total",
    "paises_unicos_visitados_ultimos_3_años",
    "paises_unicos_visitados_ultimos_5_años",
    "ciudades_unicas_ultimos_5_años",
    "ratio_paises_conciertos_5y",
    "conciertos_totales_pais_previos",
    "gap_medio_global_pais",
)


def enriquecer_country_global_features() -> dict:
    """
    Pipeline de enriquecimiento con country global features.

    1. Carga el JSONL en un DataFrame.
    2. Pre-computa vectorialmente (O(n log n)) las dos features cross-artista:
       - conciertos_totales_pais_previos: groupby("pais").cumcount() sobre el df
         ordenado por [pais, fecha_dt]. Devuelve la posición 0-based dentro del grupo,
         que equivale al número de visitas anteriores de cualquier artista a ese país.
       - gap_medio_global_pais: expanding mean de diffs de fecha dentro del grupo de
         país, desplazada un paso (shift(1)) para garantizar que el registro actual
         no se incluye en su propio cómputo (sin data leakage).
    3. Itera agrupado por artista para calcular las features centradas en el artista
       (diversidad geográfica, ratio países/conciertos) sin necesidad de filtrar el
       índice cross-artista por cada registro.
    4. Sobreescribe data/processed/conciertos_enriquecido.jsonl.

    Devuelve un dict resumen con total de registros procesados y ruta de destino.
    """
    ruta_enriquecido = Path(RUTA_ENRIQUECIDO)
    if not ruta_enriquecido.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {RUTA_ENRIQUECIDO}")

    dataframe = pd.read_json(ruta_enriquecido, lines=True, dtype=False)
    logger.info("Cargados %d registros desde %s", len(dataframe), RUTA_ENRIQUECIDO)

    dataframe["fecha_dt"] = pd.to_datetime(dataframe["fecha"], format="%Y-%m-%d", errors="coerce")
    dataframe = dataframe.sort_values(["artista", "fecha_dt"]).reset_index(drop=True)

    columnas_originales = [columna for columna in dataframe.columns if columna != "fecha_dt"]

    print(f"[4/5 Country Global] Iniciando — {len(dataframe)} registros, {dataframe['artista'].nunique()} artistas", flush=True)
    print(f"[4/5 Country Global]  Pre-computando features cross-artista por país...", flush=True)

    # Ordenar por país y fecha para que cumcount() y diff() operen cronológicamente
    df_pais = dataframe.sort_values(["pais", "fecha_dt"]).copy()

    # Número de visitas previas de cualquier artista al mismo país (0-based cumcount)
    df_pais["conciertos_totales_pais_previos"] = df_pais.groupby("pais").cumcount()

    # Gap medio entre visitas consecutivas al país hasta (sin incluir) el registro actual.
    # diff() da el gap entre cada par de visitas consecutivas dentro del grupo.
    # expanding().mean() acumula la media, shift(1) desplaza para excluir el registro actual.
    df_pais["gap_medio_global_pais"] = (
        df_pais.groupby("pais", group_keys=False)["fecha_dt"]
        .transform(lambda serie: serie.diff().dt.days.expanding().mean().shift(1))
    )

    # Unir columnas pre-computadas al dataframe principal (alineación por índice)
    dataframe = dataframe.join(df_pais[["conciertos_totales_pais_previos", "gap_medio_global_pais"]])

    print(f"[4/5 Country Global]  Pre-cómputo completado — {dataframe['pais'].nunique()} países distintos. Iniciando bucle de artistas...", flush=True)

    resultados = []
    total_sin_fecha_valida = 0
    total_artistas = dataframe["artista"].nunique()
    artistas_procesados = 0

    for nombre_artista, grupo_artista in dataframe.groupby("artista", sort=False):
        grupo_artista = grupo_artista.reset_index(drop=True)
        registros_artista = grupo_artista.to_dict(orient="records")
        artistas_procesados += 1
        if artistas_procesados % 5 == 0 or artistas_procesados == total_artistas:
            print(f"[4/5 Country Global]  {artistas_procesados}/{total_artistas} artistas — {nombre_artista} ({len(registros_artista)} conciertos) — {len(resultados)} registros acumulados", flush=True)

        for indice, registro_actual in enumerate(registros_artista):
            fecha_dt_actual = registro_actual.get("fecha_dt")

            registro_enriquecido = {
                clave: registro_actual[clave]
                for clave in columnas_originales
                if clave in registro_actual
            }

            if pd.isna(fecha_dt_actual):
                for nombre_feature in NOMBRES_FEATURES_COUNTRY_GLOBAL:
                    registro_enriquecido[nombre_feature] = None
                resultados.append(registro_enriquecido)
                total_sin_fecha_valida += 1
                continue

            fecha_actual_dt = fecha_dt_actual.to_pydatetime()
            pais_actual = registro_actual.get("pais", "")

            # Slice del artista con fecha estrictamente anterior (features centradas en artista)
            registros_previos_artista = [
                {clave: r[clave] for clave in columnas_originales if clave in r}
                for r in registros_artista[:indice]
                if not pd.isna(r.get("fecha_dt")) and r["fecha_dt"] < fecha_dt_actual
            ]

            features = calcular_features_country_global(
                registros_previos_artista, pais_actual, fecha_actual_dt
            )

            # Añadir features cross-artista obtenidas del pre-cómputo vectorial
            conciertos_pais_raw = registro_actual.get("conciertos_totales_pais_previos")
            gap_pais_raw = registro_actual.get("gap_medio_global_pais")
            # cumcount() devuelve float64 cuando hay filas con pais=null (groupby las excluye)
            features["conciertos_totales_pais_previos"] = (
                int(conciertos_pais_raw) if not pd.isna(conciertos_pais_raw) else None
            )
            features["gap_medio_global_pais"] = (
                None if (gap_pais_raw is None or pd.isna(gap_pais_raw))
                else round(float(gap_pais_raw), 4)
            )

            registro_enriquecido.update(features)
            resultados.append(registro_enriquecido)

    print(f"[4/5 Country Global]  Todos los artistas procesados. Escribiendo fichero...", flush=True)
    with open(ruta_enriquecido, "w", encoding="utf-8") as archivo_salida:
        for registro in resultados:
            linea = _convertir_nan_a_none(registro)
            archivo_salida.write(json.dumps(linea, ensure_ascii=False) + "\n")

    print(f"[4/5 Country Global] Completado — {len(resultados)} registros escritos en {RUTA_ENRIQUECIDO}", flush=True)
    resumen = {
        "total_procesados": len(resultados),
        "sin_fecha_valida": total_sin_fecha_valida,
        "archivo_destino": RUTA_ENRIQUECIDO,
    }

    logger.info(
        "Country global features completadas — procesados: %d | sin fecha válida: %d",
        len(resultados),
        total_sin_fecha_valida,
    )

    return resumen
