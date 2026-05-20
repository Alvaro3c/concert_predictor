"""Orquestador de predicción: features → modelo LightGBM → LLM → respuesta final."""

import logging

import numpy as np

from app.libraries.predict_feature_utils import obtener_features_para_inferencia
from app.libraries.predict_llm_utils import generar_explicacion_natural
from app.libraries.sanitizacion_inputs import sanitizar_nombre_artista, sanitizar_nombre_pais
from app.libraries.train_metrics_utils import UMBRAL_GAP_TRAMO_D, UMBRAL_VISITAS_TRAMO_D
from app.libraries.train_model_io import load_model
from config import UMBRAL_TRAMO_B, UMBRAL_TRAMO_C

logger = logging.getLogger(__name__)

RUTA_JSONL = "data/processed/conciertos_enriquecido.jsonl"
RUTA_MODELO = "models/model_predict/model.joblib"
DIRECTORIO_ENCODERS = "models/model_predict"


def _es_tramo_d(visitas_previas: float, gap_medio: float) -> bool:
    """Aplica la regla explícita del Tramo D para un solo registro."""
    return (visitas_previas <= UMBRAL_VISITAS_TRAMO_D) or (gap_medio > UMBRAL_GAP_TRAMO_D)


def _aplicar_umbrales(proba_por_clase: dict[str, float]) -> str:
    """
    Aplica los umbrales de decisión personalizados a las probabilidades del clasificador.

    Replica la lógica usada en entrenamiento: C → B → clase con mayor probabilidad.
    Esto aumenta el recall de B y C respecto al argmax puro.
    """
    if proba_por_clase.get("C", 0.0) >= UMBRAL_TRAMO_C:
        return "C"
    if proba_por_clase.get("B", 0.0) >= UMBRAL_TRAMO_B:
        return "B"
    return max(proba_por_clase, key=lambda tramo: proba_por_clase[tramo])


def predecir_retorno(nombre_artista: str, nombre_pais: str) -> dict:
    """
    Orquesta la predicción completa de retorno de un artista a un país.

    Pasos:
      1. Carga el historial del artista en el JSONL y computa las features de inferencia.
      2. Aplica la regla de Tramo D antes de llamar al modelo.
      3. Si no es Tramo D, carga el modelo LightGBM y obtiene probabilidades por tramo.
      4. Aplica los umbrales de decisión personalizados.
      5. Llama a Mistral para generar la explicación en lenguaje natural.

    Parámetros:
        nombre_artista: nombre del artista tal como aparece en el dataset.
        nombre_pais: nombre del país tal como aparece en el dataset.

    Retorna un dict con tramo_predicho, probabilidades, historial resumido y explicación.
    Lanza ValueError si el artista o el país no tienen historial en el dataset.
    Lanza FileNotFoundError si el modelo o los encoders no existen en disco.
    """
    nombre_artista = sanitizar_nombre_artista(nombre_artista)
    nombre_pais = sanitizar_nombre_pais(nombre_pais)

    vector_features, historial = obtener_features_para_inferencia(
        RUTA_JSONL, nombre_artista, nombre_pais, DIRECTORIO_ENCODERS
    )

    visitas_previas = float(vector_features.get("visitas_previas_al_pais", 0))
    gap_medio = float(vector_features.get("gap_medio_pais", 0))

    if _es_tramo_d(visitas_previas, gap_medio):
        tramo_final = "D"
        probabilidades: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 1.0}
        logger.info(
            "Tramo D asignado por regla explícita — artista='%s', país='%s', "
            "visitas_previas=%.0f, gap_medio=%.0f",
            nombre_artista, nombre_pais, visitas_previas, gap_medio,
        )
    else:
        modelo = load_model(RUTA_MODELO)

        # Construir el array en el mismo orden de columnas que el modelo espera
        nombres_features_modelo = modelo.feature_name_
        valores_features = np.array(
            [float(vector_features.get(nombre, 0)) for nombre in nombres_features_modelo],
            dtype=float,
        ).reshape(1, -1)

        proba_array = modelo.predict_proba(valores_features)[0]
        clases = list(modelo.classes_)

        probabilidades = {
            str(clase): round(float(prob), 4)
            for clase, prob in zip(clases, proba_array)
        }
        tramo_final = _aplicar_umbrales(probabilidades)

        logger.info(
            "Predicción del modelo — artista='%s', país='%s', tramo='%s', proba=%s",
            nombre_artista, nombre_pais, tramo_final, probabilidades,
        )

    explicacion = generar_explicacion_natural(
        nombre_artista, nombre_pais, historial, tramo_final, probabilidades
    )

    return {
        "artista": nombre_artista,
        "pais": nombre_pais,
        "tramo_predicho": tramo_final,
        "probabilidades": probabilidades,
        "ultimo_concierto_conocido": historial["ultima_visita"],
        "gap_medio_historico_dias": historial["gap_medio_dias"],
        "total_visitas_registradas": historial["total_visitas"],
        "explicacion": explicacion,
    }
