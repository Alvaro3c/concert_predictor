"""Módulo de entrenamiento del modelo de predicción de días entre conciertos."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import LabelEncoder

from app.libraries.train_data_validation import (
    COLUMNAS_FEATURES,
    COLUMNAS_MINIMAS,
    PATRONES_LEAKAGE,
    validate_date_column,
    validate_no_future_leakage,
    validate_required_columns,
)
from app.libraries.train_feature_utils import (
    compute_warmup_cutoff,
    encode_boolean_flags,
    encode_season,
)
from app.libraries.train_metrics_utils import compute_regression_metrics, log_metrics_to_file
from app.libraries.train_model_io import save_feature_importance, save_model
from app.libraries.train_split_utils import temporal_split_by_artist
from config import FECHA_FIN_COVID, FECHA_INICIO_COVID, FILTRAR_GAPS_COVID

logger = logging.getLogger(__name__)

NOMBRE_TARGET = "dias_hasta_proximo"
NOMBRE_FICHERO_MODELO = "model.joblib"
NOMBRE_FICHERO_METRICAS = "metrics.json"
NOMBRE_FICHERO_IMPORTANCIA = "feature_importance.csv"

COLUMNAS_REQUERIDAS = COLUMNAS_MINIMAS + [col for col in COLUMNAS_FEATURES if col not in COLUMNAS_MINIMAS]

COLUMNA_ARTISTA = "artista"
COLUMNA_PAIS = "pais"
COLUMNA_FECHA = "fecha"
COLUMNA_TEMPORADA = "estacion"
COLUMNA_FECHA_DT = "_fecha_dt"
COLUMNA_FECHA_SIGUIENTE_DT = "_fecha_siguiente_dt"
ANNIOS_WARMUP = 3

DIAS_MINIMO_GAP = 7
DIAS_MAXIMO_GAP = 1095

COLUMNAS_BOOLEANAS = [
    "en_gira_activa",
    "es_primera_visita_observada",
    "es_post_covid",
    "es_periodo_covid",
]

PARAMETROS_LIGHTGBM: dict = {
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "min_child_samples": 20,
    "random_state": 42,
    "verbosity": -1,
}
RONDAS_EARLY_STOPPING = 50
PERIODO_LOG = 100

# Encoders reutilizables entre entrenamiento e inferencia
_encoder_artista = LabelEncoder()
_encoder_pais = LabelEncoder()


def cargar_y_validar_dataset(ruta_jsonl: str) -> pd.DataFrame:
    """
    Lee el JSONL de conciertos enriquecidos línea a línea, valida su integridad
    y retorna un DataFrame limpio listo para feature engineering.

    Parámetros:
        ruta_jsonl: ruta al fichero JSONL con los datos enriquecidos.

    Retorna un pd.DataFrame con todos los registros válidos.
    Lanza FileNotFoundError si el fichero no existe.
    Lanza ValueError si alguna validación falla.
    """
    ruta = Path(ruta_jsonl)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_jsonl}")

    registros: list[dict] = []
    with open(ruta, "r", encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            linea_limpia = linea.strip()
            if not linea_limpia:
                continue
            try:
                registros.append(json.loads(linea_limpia))
            except json.JSONDecodeError as error:
                logger.warning("Error JSON en línea %d, se omite: %s", numero_linea, error)

    if not registros:
        raise ValueError(f"El fichero JSONL no contiene registros válidos: {ruta_jsonl}")

    dataframe = pd.DataFrame(registros)
    logger.info("Cargados %d registros desde %s", len(dataframe), ruta_jsonl)

    if not validate_required_columns(dataframe, COLUMNAS_REQUERIDAS):
        columnas_faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in dataframe.columns]
        raise ValueError(f"Columnas requeridas ausentes en el dataset: {columnas_faltantes}")

    if not validate_date_column(dataframe, "fecha"):
        raise ValueError(
            "La columna 'fecha' contiene demasiados valores inválidos (supera el umbral del 5%)"
        )

    if not validate_no_future_leakage(dataframe):
        columnas_con_leakage = [
            col for col in dataframe.columns
            if any(patron in col.lower() for patron in PATRONES_LEAKAGE)
        ]
        raise ValueError(
            f"Detectado posible data leakage del futuro en columnas: {columnas_con_leakage}"
        )

    return dataframe


def preparar_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Aplica todas las transformaciones de features y calcula el target.

    Target: días hasta el próximo concierto del mismo artista en el mismo país,
    calculado como la diferencia en días entre registros consecutivos por par (artista, país).

    Parámetros:
        dataframe: DataFrame limpio procedente de cargar_y_validar_dataset.

    Retorna una tupla (features_df, target_serie, nombres_features).
    """
    dataframe_copia = dataframe.copy()

    if COLUMNA_TEMPORADA in dataframe_copia.columns:
        dataframe_copia[COLUMNA_TEMPORADA] = (
            dataframe_copia[COLUMNA_TEMPORADA].apply(encode_season)
        )

    columnas_booleanas_presentes = [
        col for col in COLUMNAS_BOOLEANAS if col in dataframe_copia.columns
    ]
    dataframe_copia = encode_boolean_flags(dataframe_copia, columnas_booleanas_presentes)

    dataframe_copia[COLUMNA_ARTISTA] = _encoder_artista.fit_transform(
        dataframe_copia[COLUMNA_ARTISTA].astype(str)
    )
    dataframe_copia[COLUMNA_PAIS] = _encoder_pais.fit_transform(
        dataframe_copia[COLUMNA_PAIS].astype(str)
    )

    dataframe_copia[COLUMNA_FECHA_DT] = pd.to_datetime(
        dataframe_copia[COLUMNA_FECHA], errors="coerce"
    )
    dataframe_copia = dataframe_copia.sort_values(
        [COLUMNA_ARTISTA, COLUMNA_PAIS, COLUMNA_FECHA_DT]
    ).reset_index(drop=True)

    fecha_siguiente_por_par = (
        dataframe_copia.groupby([COLUMNA_ARTISTA, COLUMNA_PAIS])[COLUMNA_FECHA_DT].shift(-1)
    )
    dataframe_copia[NOMBRE_TARGET] = (
        fecha_siguiente_por_par - dataframe_copia[COLUMNA_FECHA_DT]
    ).dt.days
    dataframe_copia[COLUMNA_FECHA_SIGUIENTE_DT] = fecha_siguiente_por_par

    dataframe_copia = compute_warmup_cutoff(
        dataframe_copia, COLUMNA_ARTISTA, COLUMNA_FECHA_DT, annios_warmup=ANNIOS_WARMUP
    )
    dataframe_copia = dataframe_copia[~dataframe_copia["is_warmup"]].copy()
    dataframe_copia = dataframe_copia.drop(columns=["is_warmup"])

    filas_antes = len(dataframe_copia)
    dataframe_copia = dataframe_copia.dropna(subset=[NOMBRE_TARGET]).reset_index(drop=True)
    logger.info(
        "Eliminadas %d filas sin target (último concierto por artista/país)",
        filas_antes - len(dataframe_copia),
    )

    # Filtrar gaps no predecibles: fechas de gira consecutiva (<7 días) o hiatuses extremos (>3 años)
    filas_antes_filtro = len(dataframe_copia)
    dataframe_copia = dataframe_copia[
        (dataframe_copia[NOMBRE_TARGET] >= DIAS_MINIMO_GAP) &
        (dataframe_copia[NOMBRE_TARGET] <= DIAS_MAXIMO_GAP)
    ].reset_index(drop=True)
    logger.info(
        "Eliminadas %d filas con gaps fuera del rango [%d, %d] días",
        filas_antes_filtro - len(dataframe_copia),
        DIAS_MINIMO_GAP,
        DIAS_MAXIMO_GAP,
    )

    if FILTRAR_GAPS_COVID:
        inicio_covid = pd.Timestamp(FECHA_INICIO_COVID)
        fin_covid = pd.Timestamp(FECHA_FIN_COVID)
        filas_antes_covid = len(dataframe_copia)
        mascara_cruza_covid = (
            dataframe_copia[COLUMNA_FECHA_SIGUIENTE_DT].notna() &
            (dataframe_copia[COLUMNA_FECHA_DT] < fin_covid) &
            (dataframe_copia[COLUMNA_FECHA_SIGUIENTE_DT] > inicio_covid)
        )
        dataframe_copia = dataframe_copia[~mascara_cruza_covid].reset_index(drop=True)
        logger.info(
            "Eliminadas %d filas con gaps que atraviesan el período COVID (%s — %s)",
            filas_antes_covid - len(dataframe_copia),
            FECHA_INICIO_COVID,
            FECHA_FIN_COVID,
        )

    dataframe_copia = dataframe_copia.drop(
        columns=[COLUMNA_FECHA_DT, COLUMNA_FECHA_SIGUIENTE_DT]
    )

    # Transformar target a escala logarítmica — mejora el aprendizaje con distribuciones sesgadas
    dataframe_copia[NOMBRE_TARGET] = np.log1p(dataframe_copia[NOMBRE_TARGET])

    columnas_excluidas = {NOMBRE_TARGET, COLUMNA_FECHA}
    nombres_features = [
        col for col in COLUMNAS_FEATURES
        if col in dataframe_copia.columns and col not in columnas_excluidas
    ]

    columnas_features_df = nombres_features + (
        [COLUMNA_FECHA] if COLUMNA_FECHA in dataframe_copia.columns else []
    )
    features_df = dataframe_copia[columnas_features_df].copy()
    target_serie = dataframe_copia[NOMBRE_TARGET].copy()

    logger.info(
        "Features preparadas — %d filas, %d features de modelo",
        len(features_df),
        len(nombres_features),
    )
    return features_df, target_serie, nombres_features


def entrenar_lightgbm(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_val: pd.DataFrame,
    target_val: pd.Series,
    parametros: dict | None = None,
) -> LGBMRegressor:
    """
    Entrena un LGBMRegressor con early stopping evaluado sobre el conjunto de validación.

    Parámetros:
        features_train: features del conjunto de entrenamiento.
        target_train: target del conjunto de entrenamiento (días).
        features_val: features del conjunto de validación para early stopping.
        target_val: target del conjunto de validación para early stopping.
        parametros: dict de hiperparámetros para LGBMRegressor. Si None usa los por defecto.

    Retorna el modelo LGBMRegressor entrenado.
    """
    params_entrenamiento = {**PARAMETROS_LIGHTGBM, **(parametros or {})}

    logger.info(
        "Iniciando entrenamiento LightGBM — train: %d filas, val: %d filas",
        len(features_train),
        len(features_val),
    )

    modelo = LGBMRegressor(**params_entrenamiento)
    modelo.fit(
        features_train,
        target_train,
        eval_set=[(features_val, target_val)],
        callbacks=[
            early_stopping(stopping_rounds=RONDAS_EARLY_STOPPING, verbose=False),
            log_evaluation(period=PERIODO_LOG),
        ],
    )

    logger.info("Entrenamiento completado — mejor iteración: %d", modelo.best_iteration_)
    return modelo


def evaluar_modelo(
    modelo,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    nombres_features: list[str],
    directorio_salida: str | Path,
) -> dict:
    """
    Evalúa el modelo sobre el conjunto de test y persiste métricas e importancias.

    Las predicciones negativas se clampean a 0 (los días no pueden ser negativos).

    Parámetros:
        modelo: modelo LightGBM entrenado.
        features_test: features del conjunto de test.
        target_test: target real del conjunto de test (días).
        nombres_features: lista de nombres de features en el mismo orden que el modelo.
        directorio_salida: directorio donde persistir métricas e importancias.

    Retorna el dict de métricas calculadas (mae, rmse, r2, mape).
    """
    directorio = Path(directorio_salida)
    directorio.mkdir(parents=True, exist_ok=True)

    predicciones_raw = modelo.predict(features_test)
    # Revertir la transformación log1p para obtener días reales interpretables
    predicciones_log = np.maximum(predicciones_raw, 0)
    predicciones = np.expm1(predicciones_log)
    target_dias = np.expm1(target_test.values)

    metricas = compute_regression_metrics(target_dias, predicciones)
    log_metrics_to_file(metricas, directorio / NOMBRE_FICHERO_METRICAS)
    save_feature_importance(modelo, nombres_features, directorio / NOMBRE_FICHERO_IMPORTANCIA)

    logger.info(
        "Evaluación completada — MAE: %.1f días | RMSE: %.1f días | R²: %.4f | MAPE: %s%%",
        metricas["mae"],
        metricas["rmse"],
        metricas["r2"],
        f"{metricas['mape']:.2f}" if metricas["mape"] is not None else "N/A",
    )
    return metricas


def train_pipeline(ruta_jsonl: str, directorio_salida: str) -> dict:
    """
    Orquesta el pipeline completo de entrenamiento del modelo LightGBM.

    Pasos:
      1. cargar_y_validar_dataset  → carga y valida el JSONL.
      2. preparar_features         → codifica features y calcula el target.
      3. temporal_split_by_artist  → split cronológico 80/20 por artista.
      4. entrenar_lightgbm         → entrena con early stopping.
      5. evaluar_modelo            → predice, calcula métricas y guarda resultados.
      6. save_model                → persiste el modelo en disco.

    Parámetros:
        ruta_jsonl: ruta al fichero JSONL de entrada.
        directorio_salida: directorio donde guardar el modelo y las métricas.

    Retorna un dict con las métricas de evaluación y la ruta del modelo guardado.
    """
    print(f"[1/6] Cargando dataset: {ruta_jsonl}", flush=True)
    dataframe = cargar_y_validar_dataset(ruta_jsonl)
    print(f"[1/6] Dataset cargado — {len(dataframe)} registros, {dataframe['artista'].nunique()} artistas", flush=True)

    print("[2/6] Calculando features y target...", flush=True)
    features_df, target_serie, nombres_features = preparar_features(dataframe)
    print(f"[2/6] Features listas — {len(features_df)} filas tras warmup, {len(nombres_features)} features", flush=True)

    print("[3/6] Dividiendo train/test temporalmente por artista...", flush=True)
    df_para_split = features_df.assign(**{NOMBRE_TARGET: target_serie.values})
    train_df, test_df = temporal_split_by_artist(
        df_para_split, COLUMNA_ARTISTA, COLUMNA_FECHA, ratio_train=0.8
    )

    if len(test_df) == 0:
        raise ValueError(
            "El conjunto de test quedó vacío tras el split. "
            "Se necesitan más datos o artistas con más de un concierto post-warmup."
        )
    print(f"[3/6] Split completado — Train: {len(train_df)} filas | Test: {len(test_df)} filas", flush=True)

    features_train = train_df[nombres_features]
    target_train = train_df[NOMBRE_TARGET]
    features_test = test_df[nombres_features]
    target_test = test_df[NOMBRE_TARGET]

    print("[4/6] Entrenando LightGBM con early stopping...", flush=True)
    modelo = entrenar_lightgbm(features_train, target_train, features_test, target_test)
    print(f"[4/6] Entrenamiento completado — mejor iteración: {modelo.best_iteration_}", flush=True)

    print("[5/6] Evaluando modelo y guardando métricas...", flush=True)
    metricas = evaluar_modelo(modelo, features_test, target_test, nombres_features, directorio_salida)
    print(f"[5/6] MAE: {metricas['mae']:.1f} días | RMSE: {metricas['rmse']:.1f} | R²: {metricas['r2']:.4f}", flush=True)

    print("[6/6] Guardando modelo en disco...", flush=True)
    ruta_modelo = Path(directorio_salida) / NOMBRE_FICHERO_MODELO
    save_model(modelo, ruta_modelo)
    print(f"[6/6] Modelo guardado en {ruta_modelo}", flush=True)

    return {"metricas": metricas, "ruta_modelo": str(ruta_modelo)}
