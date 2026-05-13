"""Funciones de cálculo y persistencia de métricas de regresión."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
