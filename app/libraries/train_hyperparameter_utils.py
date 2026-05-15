"""Búsqueda automática de hiperparámetros con Optuna para LightGBM clasificador."""

import logging

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


def buscar_hiperparametros(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_val: pd.DataFrame,
    target_val: pd.Series,
    num_trials: int,
) -> dict:
    """
    Busca los mejores hiperparámetros para LGBMClassifier usando Optuna.

    Entrena un modelo temporal en cada trial con distintos valores de
    hiperparámetros y lo evalúa en el conjunto de validación midiendo
    el F1 macro sobre los tramos A, B y C (excluye D, que es una regla).

    Parámetros:
        features_train: matriz de features de entrenamiento.
        target_train: etiquetas de tramo para entrenamiento (A/B/C, sin D).
        features_val: matriz de features de validación.
        target_val: etiquetas de tramo para validación (A/B/C, sin D).
        num_trials: número de combinaciones que Optuna va a probar.

    Retorna:
        dict con los mejores hiperparámetros encontrados.
    """

    def objetivo(trial: optuna.Trial) -> float:
        parametros = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }

        modelo = LGBMClassifier(**parametros)
        modelo.fit(features_train, target_train)
        predicciones = modelo.predict(features_val)

        return f1_score(
            target_val,
            predicciones,
            labels=["A", "B", "C"],
            average="macro",
            zero_division=0,
        )

    estudio = optuna.create_study(direction="maximize")
    estudio.optimize(objetivo, n_trials=num_trials, show_progress_bar=False)

    mejores_params = estudio.best_params
    logger.info(
        "Optuna completado en %d trials. Mejor F1 macro: %.4f. Params: %s",
        num_trials,
        estudio.best_value,
        mejores_params,
    )

    return mejores_params
