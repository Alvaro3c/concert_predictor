"""Funciones de cálculo y persistencia de métricas de regresión y clasificación por tramos."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LIMITE_TRAMO_A = 365
LIMITE_TRAMO_B = 730
LIMITE_TRAMO_C = 1460
UMBRAL_VISITAS_TRAMO_D = 0
UMBRAL_GAP_TRAMO_D = 1460

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    valores_reales: np.ndarray,
    valores_predichos: np.ndarray,
) -> dict:
    """
    Calcula métricas estándar de regresión: MAE, RMSE, R² y MAPE.

    El MAPE se calcula solo sobre observaciones donde el valor real es distinto
    de cero para evitar división por cero. Si todos los valores reales son cero,
    se devuelve None para mape.

    Parámetros:
        valores_reales: array de valores objetivo reales.
        valores_predichos: array de predicciones del modelo.

    Retorna un dict con las claves: mae, rmse, r2, mape.
    """
    mae = float(mean_absolute_error(valores_reales, valores_predichos))
    rmse = float(np.sqrt(mean_squared_error(valores_reales, valores_predichos)))
    r2 = float(r2_score(valores_reales, valores_predichos))

    mascara_no_cero = valores_reales != 0
    if mascara_no_cero.sum() == 0:
        logger.warning("Todos los valores reales son cero, MAPE no puede calcularse")
        mape = None
    else:
        mape = float(
            np.mean(
                np.abs(
                    (valores_reales[mascara_no_cero] - valores_predichos[mascara_no_cero])
                    / valores_reales[mascara_no_cero]
                )
            )
            * 100
        )

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def log_metrics_to_file(metricas: dict, ruta_salida: Path | str) -> None:
    """
    Persiste el dict de métricas en un fichero JSON con timestamp UTC.

    Parámetros:
        metricas: dict con las métricas calculadas.
        ruta_salida: ruta del fichero JSON de destino.
    """
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metricas": metricas,
    }

    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, ensure_ascii=False)

    logger.info("Métricas persistidas en %s", ruta)


def construir_mascara_tramo_d(
    visitas_previas: np.ndarray,
    gap_medio: np.ndarray,
) -> np.ndarray:
    """
    Devuelve máscara booleana donde True indica que aplica Tramo D.

    Condiciones (cualquiera activa el Tramo D):
        - visitas_previas <= 1: nunca ha tocado o solo una vez en ese país
        - gap_medio > UMBRAL_GAP_TRAMO_D: histórico medio mayor de 4 años
    """
    sin_historial_suficiente = np.array(visitas_previas, dtype=float) <= UMBRAL_VISITAS_TRAMO_D
    gap_serie = pd.array(gap_medio, dtype="Float64")
    gap_extremo = np.array((gap_serie > UMBRAL_GAP_TRAMO_D).fillna(False), dtype=bool)
    return sin_historial_suficiente | gap_extremo


def _asignar_tramo(dias: float, es_tramo_d: bool) -> str:
    """Asigna el tramo (A/B/C/D) a un único registro."""
    if es_tramo_d:
        return "D"
    if dias < LIMITE_TRAMO_A:
        return "A"
    if dias < LIMITE_TRAMO_B:
        return "B"
    return "C"


def compute_bucket_metrics(
    target_dias: np.ndarray,
    predicciones_dias: np.ndarray,
    visitas_previas: np.ndarray,
    gap_medio_pais: np.ndarray,
) -> dict:
    """
    Calcula métricas de clasificación por tramos temporales.

    El Tramo D se asigna por regla explícita antes de evaluar la predicción en días.
    Se aplica igual a target real y a predicciones, por lo que los registros D
    siempre computan como acierto (la regla es determinista).

    Tramos:
        A: < 365 días   (menos de 1 año)
        B: 365-730 días (1-2 años)
        C: 730-1460 días (2-4 años, aunque el modelo filtra >1095 en training)
        D: regla explícita — sin historial suficiente o gap histórico extremo

    Parámetros:
        target_dias: días reales (ya revertidos de log1p).
        predicciones_dias: días predichos por el modelo (ya revertidos de log1p).
        visitas_previas: columna visitas_previas_al_pais del test set.
        gap_medio_pais: columna gap_medio_pais del test set.

    Retorna un dict con bucket_accuracy_global, bucket_accuracy_por_tramo,
    recall_tramo_a y distribucion_real.
    """
    mascara_d = construir_mascara_tramo_d(visitas_previas, gap_medio_pais)

    tramos_reales = [
        _asignar_tramo(dias, es_d)
        for dias, es_d in zip(target_dias, mascara_d)
    ]
    tramos_predichos = [
        _asignar_tramo(dias, es_d)
        for dias, es_d in zip(predicciones_dias, mascara_d)
    ]

    total = len(tramos_reales)
    aciertos_global = sum(real == pred for real, pred in zip(tramos_reales, tramos_predichos))
    bucket_accuracy_global = round(aciertos_global / total, 4) if total > 0 else None

    bucket_accuracy_por_tramo: dict = {}
    distribucion_real: dict = {}

    for tramo in ["A", "B", "C", "D"]:
        indices_tramo = [indice for indice, real in enumerate(tramos_reales) if real == tramo]
        distribucion_real[tramo] = len(indices_tramo)

        if indices_tramo:
            aciertos_tramo = sum(1 for indice in indices_tramo if tramos_predichos[indice] == tramo)
            bucket_accuracy_por_tramo[tramo] = round(aciertos_tramo / len(indices_tramo), 4)
        else:
            bucket_accuracy_por_tramo[tramo] = None

    recall_tramo_a = bucket_accuracy_por_tramo["A"]

    return {
        "bucket_accuracy_global": bucket_accuracy_global,
        "bucket_accuracy_por_tramo": bucket_accuracy_por_tramo,
        "recall_tramo_a": recall_tramo_a,
        "distribucion_real": distribucion_real,
    }


def compute_classification_metrics(
    target_tramos: np.ndarray,
    predicciones_tramos: np.ndarray,
) -> dict:
    """
    Calcula métricas de clasificación por tramos A, B, C, D.

    Para cada tramo calcula:
        - Recall (bucket_accuracy_por_tramo): de los casos reales de ese tramo,
          qué porcentaje predijo correctamente.
        - Precision: de los casos predichos como ese tramo, qué porcentaje era correcto.
        - F1: media armónica de precision y recall.

    Parámetros:
        target_tramos: etiquetas reales ("A", "B", "C", "D").
        predicciones_tramos: etiquetas predichas por el modelo con Tramo D ya aplicado.

    Retorna un dict con bucket_accuracy_global, bucket_accuracy_por_tramo,
    precision_por_tramo, f1_por_tramo, recall_tramo_a y distribucion_real.
    """
    total = len(target_tramos)
    aciertos_global = sum(real == pred for real, pred in zip(target_tramos, predicciones_tramos))
    bucket_accuracy_global = round(aciertos_global / total, 4) if total > 0 else None

    bucket_accuracy_por_tramo: dict = {}
    precision_por_tramo: dict = {}
    f1_por_tramo: dict = {}
    distribucion_real: dict = {}

    for tramo in ["A", "B", "C", "D"]:
        indices_reales = [indice for indice, real in enumerate(target_tramos) if real == tramo]
        indices_predichos = [indice for indice, pred in enumerate(predicciones_tramos) if pred == tramo]

        distribucion_real[tramo] = len(indices_reales)

        if indices_reales:
            aciertos_recall = sum(1 for indice in indices_reales if predicciones_tramos[indice] == tramo)
            bucket_accuracy_por_tramo[tramo] = round(aciertos_recall / len(indices_reales), 4)
        else:
            bucket_accuracy_por_tramo[tramo] = None

        if indices_predichos:
            aciertos_precision = sum(1 for indice in indices_predichos if target_tramos[indice] == tramo)
            precision_por_tramo[tramo] = round(aciertos_precision / len(indices_predichos), 4)
        else:
            precision_por_tramo[tramo] = None

        recall = bucket_accuracy_por_tramo[tramo]
        precision = precision_por_tramo[tramo]
        if recall is not None and precision is not None and (recall + precision) > 0:
            f1_por_tramo[tramo] = round(2 * precision * recall / (precision + recall), 4)
        else:
            f1_por_tramo[tramo] = None

    return {
        "bucket_accuracy_global": bucket_accuracy_global,
        "bucket_accuracy_por_tramo": bucket_accuracy_por_tramo,
        "precision_por_tramo": precision_por_tramo,
        "f1_por_tramo": f1_por_tramo,
        "recall_tramo_a": bucket_accuracy_por_tramo["A"],
        "distribucion_real": distribucion_real,
    }
