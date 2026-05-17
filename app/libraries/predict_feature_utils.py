"""Funciones para construir el vector de features y el historial de un artista/país en inferencia."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.libraries.date_utils import obtener_estacion
from app.libraries.train_data_validation import COLUMNAS_FEATURES
from app.libraries.train_feature_utils import encode_season
from app.libraries.train_model_io import load_encoders

logger = logging.getLogger(__name__)

DIAS_UN_AÑO = 365
DIAS_TRES_AÑOS = 1095
DIAS_CINCO_AÑOS = 1825
DIAS_TREINTA = 30
DIAS_NOVENTA = 90
UMBRAL_GIRA_ACTIVA = 3


def cargar_historial_artista(ruta_jsonl: str, nombre_artista: str) -> pd.DataFrame:
    """
    Carga todos los conciertos del artista desde el JSONL enriquecido.
    La búsqueda es insensible a mayúsculas/minúsculas.

    Lanza FileNotFoundError si el fichero no existe.
    Lanza ValueError si el artista no aparece en el dataset.
    """
    ruta = Path(ruta_jsonl)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el fichero JSONL en: {ruta_jsonl}")

    registros: list[dict] = []
    with open(ruta, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            linea_limpia = linea.strip()
            if not linea_limpia:
                continue
            try:
                registro = json.loads(linea_limpia)
                if str(registro.get("artista", "")).lower() == nombre_artista.lower():
                    registros.append(registro)
            except json.JSONDecodeError:
                continue

    if not registros:
        raise ValueError(f"No se encontró el artista '{nombre_artista}' en el dataset")

    dataframe = pd.DataFrame(registros)
    dataframe["fecha"] = pd.to_datetime(dataframe["fecha"], errors="coerce")
    dataframe = dataframe.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    logger.info("Artista '%s' — %d conciertos cargados", nombre_artista, len(dataframe))
    return dataframe


def extraer_conciertos_en_pais(dataframe_artista: pd.DataFrame, nombre_pais: str) -> pd.DataFrame:
    """
    Filtra los conciertos del artista en el país indicado (insensible a mayúsculas).

    Lanza ValueError si el artista no tiene conciertos en ese país.
    """
    mascara_pais = dataframe_artista["pais"].str.lower() == nombre_pais.lower()
    dataframe_pais = dataframe_artista[mascara_pais].copy().reset_index(drop=True)

    if dataframe_pais.empty:
        raise ValueError(
            f"No se encontraron conciertos del artista en '{nombre_pais}'"
        )

    logger.info("País '%s' — %d conciertos encontrados", nombre_pais, len(dataframe_pais))
    return dataframe_pais


def _conciertos_en_ventana(
    dataframe: pd.DataFrame,
    fecha_referencia: datetime,
    dias: int,
) -> int:
    """Cuenta conciertos dentro de los últimos `dias` días antes de fecha_referencia."""
    fecha_inicio = fecha_referencia - pd.Timedelta(days=dias)
    return int(
        ((dataframe["fecha"] >= fecha_inicio) & (dataframe["fecha"] < fecha_referencia)).sum()
    )


def calcular_features_ventana_global(
    dataframe_artista: pd.DataFrame,
    fecha_referencia: datetime,
) -> dict:
    """
    Calcula las features de ventana temporal global del artista respecto a fecha_referencia.

    Recomputa las features de actividad que en entrenamiento se calculaban por fecha de concierto,
    pero que en inferencia deben calcularse respecto al día de hoy.
    """
    conciertos_1_año = _conciertos_en_ventana(dataframe_artista, fecha_referencia, DIAS_UN_AÑO)
    conciertos_3_años = _conciertos_en_ventana(dataframe_artista, fecha_referencia, DIAS_TRES_AÑOS)
    conciertos_5_años = _conciertos_en_ventana(dataframe_artista, fecha_referencia, DIAS_CINCO_AÑOS)

    media_anual_5_años = conciertos_5_años / 5 if conciertos_5_años > 0 else 0
    tendencia_actividad = (conciertos_1_año / media_anual_5_años) if media_anual_5_años > 0 else 0.0

    # ratio_expansion_artista: media anual de últimos 2 años vs media anual de los 3 años anteriores
    conciertos_2_años = _conciertos_en_ventana(dataframe_artista, fecha_referencia, DIAS_UN_AÑO * 2)
    fecha_hace_2_años = fecha_referencia - pd.Timedelta(days=DIAS_UN_AÑO * 2)
    fecha_hace_5_años = fecha_referencia - pd.Timedelta(days=DIAS_CINCO_AÑOS)
    conciertos_periodo_anterior = int(
        (
            (dataframe_artista["fecha"] >= fecha_hace_5_años)
            & (dataframe_artista["fecha"] < fecha_hace_2_años)
        ).sum()
    )
    media_periodo_anterior = conciertos_periodo_anterior / 3 if conciertos_periodo_anterior > 0 else 0
    media_reciente = conciertos_2_años / 2 if conciertos_2_años > 0 else 0
    ratio_expansion = (media_reciente / media_periodo_anterior) if media_periodo_anterior > 0 else 0.0

    primera_fecha = dataframe_artista["fecha"].min()
    años_observados = (fecha_referencia - primera_fecha).days / 365.25

    fecha_hace_3_años = fecha_referencia - pd.Timedelta(days=DIAS_TRES_AÑOS)
    conciertos_en_5_años = dataframe_artista[
        (dataframe_artista["fecha"] >= fecha_hace_5_años)
        & (dataframe_artista["fecha"] < fecha_referencia)
    ]
    conciertos_en_3_años = dataframe_artista[
        (dataframe_artista["fecha"] >= fecha_hace_3_años)
        & (dataframe_artista["fecha"] < fecha_referencia)
    ]

    paises_unicos_total = int(dataframe_artista["pais"].nunique())
    paises_unicos_3_años = int(conciertos_en_3_años["pais"].nunique())
    paises_unicos_5_años = int(conciertos_en_5_años["pais"].nunique())

    columna_ciudad = "ciudad" if "ciudad" in dataframe_artista.columns else None
    ciudades_5_años = (
        int(conciertos_en_5_años[columna_ciudad].nunique()) if columna_ciudad else 0
    )
    ratio_paises_conciertos_5y = (
        paises_unicos_5_años / conciertos_5_años if conciertos_5_años > 0 else 0.0
    )

    ultima_fecha_global = dataframe_artista["fecha"].max()
    dias_desde_ultimo_global = (fecha_referencia - ultima_fecha_global).days

    return {
        "conciertos_ultimo_año": conciertos_1_año,
        "conciertos_ultimos_3_años": conciertos_3_años,
        "conciertos_ultimos_5_años": conciertos_5_años,
        "dias_desde_ultimo_concierto_global": dias_desde_ultimo_global,
        "tendencia_actividad": tendencia_actividad,
        "ratio_expansion_artista": ratio_expansion,
        "años_observados_artista": años_observados,
        "paises_unicos_visitados_total": paises_unicos_total,
        "paises_unicos_visitados_ultimos_3_años": paises_unicos_3_años,
        "paises_unicos_visitados_ultimos_5_años": paises_unicos_5_años,
        "ciudades_unicas_ultimos_5_años": ciudades_5_años,
        "ratio_paises_conciertos_5y": ratio_paises_conciertos_5y,
    }


def calcular_features_ventana_tour(
    dataframe_artista: pd.DataFrame,
    fecha_referencia: datetime,
) -> dict:
    """
    Calcula las features de actividad de gira reciente respecto a fecha_referencia.
    """
    conciertos_30_dias = _conciertos_en_ventana(dataframe_artista, fecha_referencia, DIAS_TREINTA)
    conciertos_90_dias = _conciertos_en_ventana(dataframe_artista, fecha_referencia, DIAS_NOVENTA)
    en_gira_activa = int(conciertos_30_dias >= UMBRAL_GIRA_ACTIVA)

    fecha_hace_30_dias = fecha_referencia - pd.Timedelta(days=DIAS_TREINTA)
    conciertos_recientes = dataframe_artista[
        (dataframe_artista["fecha"] >= fecha_hace_30_dias)
        & (dataframe_artista["fecha"] < fecha_referencia)
    ]
    paises_distintos_30_dias = int(conciertos_recientes["pais"].nunique())

    conciertos_anteriores = dataframe_artista[dataframe_artista["fecha"] < fecha_referencia]
    if conciertos_anteriores.empty:
        dias_desde_anterior = 0
    else:
        ultima_fecha_global = conciertos_anteriores["fecha"].max()
        dias_desde_anterior = (fecha_referencia - ultima_fecha_global).days

    return {
        "conciertos_artista_ultimos_30_dias": conciertos_30_dias,
        "conciertos_artista_ultimos_90_dias": conciertos_90_dias,
        "en_gira_activa": en_gira_activa,
        "dias_desde_concierto_anterior": dias_desde_anterior,
        "paises_distintos_ultimos_30_dias": paises_distintos_30_dias,
    }


def construir_vector_features(
    ultima_fila: dict,
    features_global: dict,
    features_tour: dict,
    encoder_artista,
    encoder_pais,
    nombre_artista: str,
    nombre_pais: str,
    fecha_hoy: datetime,
) -> dict:
    """
    Combina todas las fuentes de features en el vector final listo para el modelo LightGBM.

    Las features de ventana se toman de features_global y features_tour (recomputadas a hoy).
    Las features estáticas de país se toman de la última fila conocida del JSONL.
    Las features temporales (dias_desde_*, estacion, covid) se actualizan con fecha_hoy.
    artista y pais se re-encodan con los LabelEncoders persistidos en entrenamiento.

    Retorna un dict ordenado según COLUMNAS_FEATURES.
    """
    ultima_fecha_pais = pd.to_datetime(ultima_fila.get("fecha"))
    dias_desde_ultima_visita = (fecha_hoy - ultima_fecha_pais).days

    # En inferencia, ultimo_gap_pais = días transcurridos desde la última visita
    ultimo_gap = dias_desde_ultima_visita
    gap_medio = float(ultima_fila.get("gap_medio_pais") or 0)
    tendencia_gap = (ultimo_gap / gap_medio) if gap_medio > 0 else 0.0

    estacion_hoy = obtener_estacion(fecha_hoy)

    clases_artista = list(encoder_artista.classes_)
    clases_pais = list(encoder_pais.classes_)
    artista_encoded = (
        int(encoder_artista.transform([nombre_artista])[0])
        if nombre_artista in clases_artista
        else -1
    )
    pais_encoded = (
        int(encoder_pais.transform([nombre_pais])[0])
        if nombre_pais in clases_pais
        else -1
    )

    vector = {
        **features_global,
        **features_tour,
        # visitas_previas_al_pais: el último concierto conocido ya cuenta como visita previa
        "visitas_previas_al_pais": int(ultima_fila.get("visitas_previas_al_pais") or 0) + 1,
        "es_primera_visita_observada": 0,
        "dias_desde_ultima_visita_pais": dias_desde_ultima_visita,
        "gap_medio_pais": gap_medio,
        "gap_mediano_pais": float(ultima_fila.get("gap_mediano_pais") or 0),
        "gap_std_pais": float(ultima_fila.get("gap_std_pais") or 0),
        "gap_min_pais": float(ultima_fila.get("gap_min_pais") or 0),
        "gap_max_pais": float(ultima_fila.get("gap_max_pais") or 0),
        "ultimo_gap_pais": ultimo_gap,
        "tendencia_gap_pais": tendencia_gap,
        "gap_coeficiente_variacion_pais": float(ultima_fila.get("gap_coeficiente_variacion_pais") or 0),
        "ratio_visitas_misma_estacion_pais": float(ultima_fila.get("ratio_visitas_misma_estacion_pais") or 0),
        "concentracion_estacional_pais": float(ultima_fila.get("concentracion_estacional_pais") or 0),
        "proporcion_visitas_pais": float(ultima_fila.get("proporcion_visitas_pais") or 0),
        "rank_pais_para_artista": int(ultima_fila.get("rank_pais_para_artista") or 0),
        "conciertos_totales_pais_previos": int(ultima_fila.get("conciertos_totales_pais_previos") or 0),
        "gap_medio_global_pais": float(ultima_fila.get("gap_medio_global_pais") or 0),
        "estacion": encode_season(estacion_hoy),
        "es_post_covid": 1,
        "es_periodo_covid": 0,
        "artista": artista_encoded,
        "pais": pais_encoded,
    }

    return {columna: vector[columna] for columna in COLUMNAS_FEATURES if columna in vector}


def construir_historial_para_prompt(
    dataframe_pais: pd.DataFrame,
    dataframe_artista: pd.DataFrame,
    features_tour: dict,
    fecha_hoy: datetime,
) -> dict:
    """
    Construye un dict con el historial detallado del artista en el país para alimentar al LLM.

    Incluye lista de conciertos pasados, gaps entre visitas, estadísticas y actividad reciente global.
    """
    conciertos_pasados = []
    for _, fila in dataframe_pais.iterrows():
        entrada: dict = {"fecha": fila["fecha"].strftime("%Y-%m-%d")}
        if "ciudad" in fila and pd.notna(fila.get("ciudad")):
            entrada["ciudad"] = fila["ciudad"]
        if "venue" in fila and pd.notna(fila.get("venue")):
            entrada["venue"] = fila["venue"]
        conciertos_pasados.append(entrada)

    fechas_pais = dataframe_pais["fecha"].tolist()
    gaps_entre_visitas = [
        (fechas_pais[indice + 1] - fechas_pais[indice]).days
        for indice in range(len(fechas_pais) - 1)
    ]

    distribucion_estacional: dict[str, int] = {}
    for fecha_concierto in fechas_pais:
        estacion = obtener_estacion(fecha_concierto)
        distribucion_estacional[estacion] = distribucion_estacional.get(estacion, 0) + 1

    ultima_fecha_pais = dataframe_pais["fecha"].max()
    dias_desde_ultima_visita = (fecha_hoy - ultima_fecha_pais).days

    return {
        "total_visitas": len(dataframe_pais),
        "primera_visita": fechas_pais[0].strftime("%Y-%m-%d"),
        "ultima_visita": ultima_fecha_pais.strftime("%Y-%m-%d"),
        "dias_desde_ultima_visita": dias_desde_ultima_visita,
        "conciertos_pasados": conciertos_pasados,
        "gaps_entre_visitas_dias": gaps_entre_visitas,
        "gap_medio_dias": round(sum(gaps_entre_visitas) / len(gaps_entre_visitas)) if gaps_entre_visitas else 0,
        "gap_minimo_dias": min(gaps_entre_visitas) if gaps_entre_visitas else 0,
        "gap_maximo_dias": max(gaps_entre_visitas) if gaps_entre_visitas else 0,
        "distribucion_estacional": distribucion_estacional,
        "en_gira_activa": bool(features_tour["en_gira_activa"]),
        "conciertos_ultimos_30_dias": features_tour["conciertos_artista_ultimos_30_dias"],
        "conciertos_ultimos_90_dias": features_tour["conciertos_artista_ultimos_90_dias"],
    }


def obtener_features_para_inferencia(
    ruta_jsonl: str,
    nombre_artista: str,
    nombre_pais: str,
    ruta_encoders: str,
) -> tuple[dict, dict]:
    """
    Función principal de inferencia: orquesta la carga de datos, el cómputo de
    features y la construcción del historial para el prompt del LLM.

    Parámetros:
        ruta_jsonl: ruta al fichero conciertos_enriquecido.jsonl.
        nombre_artista: nombre del artista tal como aparece en el dataset.
        nombre_pais: nombre del país tal como aparece en el dataset.
        ruta_encoders: directorio donde están artista_encoder.joblib y pais_encoder.joblib.

    Retorna una tupla (vector_features, historial_para_prompt).
    """
    fecha_hoy = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    encoder_artista, encoder_pais = load_encoders(ruta_encoders)
    dataframe_artista = cargar_historial_artista(ruta_jsonl, nombre_artista)
    dataframe_pais = extraer_conciertos_en_pais(dataframe_artista, nombre_pais)

    features_global = calcular_features_ventana_global(dataframe_artista, fecha_hoy)
    features_tour = calcular_features_ventana_tour(dataframe_artista, fecha_hoy)

    ultima_fila = dataframe_pais.iloc[-1].to_dict()

    vector_features = construir_vector_features(
        ultima_fila, features_global, features_tour,
        encoder_artista, encoder_pais,
        nombre_artista, nombre_pais, fecha_hoy,
    )
    historial_para_prompt = construir_historial_para_prompt(
        dataframe_pais, dataframe_artista, features_tour, fecha_hoy,
    )

    return vector_features, historial_para_prompt
