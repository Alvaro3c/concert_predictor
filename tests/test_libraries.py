"""Tests de las funciones de library del módulo train/model_predict.

Datos sintéticos: 10 filas, 3 artistas, fechas entre 2015 y 2023.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.train.model_predict.libraries.data_validation import (
    COLUMNAS_FEATURES,
    COLUMNAS_MINIMAS,
    validate_date_column,
    validate_no_future_leakage,
    validate_required_columns,
)
from app.train.model_predict.libraries.feature_utils import (
    compute_warmup_cutoff,
    encode_boolean_flags,
    encode_season,
)
from app.train.model_predict.libraries.metrics_utils import (
    compute_regression_metrics,
    log_metrics_to_file,
)
from app.train.model_predict.libraries.model_io import (
    load_model,
    save_feature_importance,
    save_model,
)
from app.train.model_predict.libraries.split_utils import temporal_split_by_artist


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FECHAS_SINTETICAS = [
    "2015-03-01", "2016-06-15", "2020-09-20", "2022-11-10",  # ArtistA (4 conciertos)
    "2014-01-05", "2016-07-22", "2021-03-14",                 # ArtistB (3 conciertos)
    "2015-06-01", "2019-01-15", "2023-08-20",                 # ArtistC (3 conciertos)
]

ARTISTAS_SINTETICOS = (
    ["ArtistA"] * 4 + ["ArtistB"] * 3 + ["ArtistC"] * 3
)


def crear_df_base() -> pd.DataFrame:
    """DataFrame mínimo con las columnas de metadata."""
    return pd.DataFrame(
        {
            "artista": ARTISTAS_SINTETICOS,
            "pais": ["ES", "US", "ES", "US", "ES", "FR", "ES", "US", "FR", "ES"],
            "fecha": FECHAS_SINTETICAS,
            "estacion": ["primavera", "verano", "otoño", None, "invierno", "verano",
                         "primavera", "invierno", "verano", "otoño"],
            "en_gira_activa": [True, False, True, True, False, True, False, True, False, True],
            "es_primera_visita_observada": [True, True, False, False, True, True, False, True, False, False],
            "es_post_covid": [False, False, False, True, False, False, True, True, True, True],
            "es_periodo_covid": [False, False, False, True, False, False, False, False, False, False],
            "años_observados_artista": [1, 2, 6, 8, 1, 3, 8, 1, 5, 9],
        }
    )


def crear_df_completo() -> pd.DataFrame:
    """DataFrame con todas las columnas requeridas (features numéricas a cero)."""
    df_base = crear_df_base()
    columnas_extra = [
        col for col in COLUMNAS_FEATURES
        if col not in df_base.columns and col not in COLUMNAS_MINIMAS
    ]
    for columna in columnas_extra:
        df_base[columna] = 0.0
    return df_base


# ---------------------------------------------------------------------------
# Tests: data_validation
# ---------------------------------------------------------------------------

class TestValidateRequiredColumns:
    def test_todas_las_columnas_presentes_retorna_true(self):
        dataframe = crear_df_completo()
        columnas_requeridas = COLUMNAS_MINIMAS + [
            col for col in COLUMNAS_FEATURES if col not in COLUMNAS_MINIMAS
        ]
        assert validate_required_columns(dataframe, columnas_requeridas) is True

    def test_columna_faltante_retorna_false(self):
        dataframe = crear_df_base()
        assert validate_required_columns(dataframe, ["artista", "columna_inexistente"]) is False

    def test_lista_vacia_retorna_true(self):
        dataframe = crear_df_base()
        assert validate_required_columns(dataframe, []) is True


class TestValidateDateColumn:
    def test_fechas_validas_retorna_true(self):
        dataframe = crear_df_base()
        assert validate_date_column(dataframe, "fecha") is True

    def test_columna_inexistente_retorna_false(self):
        dataframe = crear_df_base()
        assert validate_date_column(dataframe, "columna_que_no_existe") is False

    def test_mas_del_cinco_porciento_nat_retorna_false(self):
        dataframe = crear_df_base()
        # 4 de 10 filas con fecha inválida → 40% NaT → debe fallar
        dataframe.loc[[0, 1, 2, 3], "fecha"] = "no-es-una-fecha"
        assert validate_date_column(dataframe, "fecha") is False

    def test_dataframe_vacio_retorna_false(self):
        dataframe = pd.DataFrame({"fecha": []})
        assert validate_date_column(dataframe, "fecha") is False


class TestValidateNoFutureleakage:
    def test_sin_columnas_sospechosas_retorna_true(self):
        dataframe = crear_df_base()
        assert validate_no_future_leakage(dataframe) is True

    def test_columna_con_next_retorna_false(self):
        dataframe = crear_df_base()
        dataframe["next_concert_date"] = 0
        assert validate_no_future_leakage(dataframe) is False

    def test_columna_con_futuro_retorna_false(self):
        dataframe = crear_df_base()
        dataframe["precio_futuro"] = 0
        assert validate_no_future_leakage(dataframe) is False

    def test_columna_con_shift_retorna_false(self):
        dataframe = crear_df_base()
        dataframe["valor_shift_1"] = 0
        assert validate_no_future_leakage(dataframe) is False


# ---------------------------------------------------------------------------
# Tests: feature_utils
# ---------------------------------------------------------------------------

class TestEncodeSeason:
    def test_mapeo_primavera(self):
        assert encode_season("primavera") == 0

    def test_mapeo_verano(self):
        assert encode_season("verano") == 1

    def test_mapeo_otono(self):
        assert encode_season("otoño") == 2

    def test_mapeo_invierno(self):
        assert encode_season("invierno") == 3

    def test_nan_retorna_menos_uno(self):
        assert encode_season(float("nan")) == -1

    def test_none_retorna_menos_uno(self):
        assert encode_season(None) == -1

    def test_valor_desconocido_retorna_menos_uno(self):
        assert encode_season("spring") == -1

    def test_mayusculas_normalizadas(self):
        assert encode_season("VERANO") == 1


class TestEncodeBooleanFlags:
    def test_true_se_convierte_a_uno(self):
        dataframe = pd.DataFrame({"en_gira_activa": [True, False, True]})
        resultado = encode_boolean_flags(dataframe, ["en_gira_activa"])
        assert list(resultado["en_gira_activa"]) == [1, 0, 1]

    def test_columnas_no_listadas_no_se_modifican(self):
        dataframe = pd.DataFrame({"en_gira_activa": [True], "otra_columna": ["texto"]})
        resultado = encode_boolean_flags(dataframe, ["en_gira_activa"])
        assert resultado["otra_columna"].iloc[0] == "texto"

    def test_columna_inexistente_no_lanza_excepcion(self):
        dataframe = pd.DataFrame({"en_gira_activa": [True]})
        resultado = encode_boolean_flags(dataframe, ["columna_que_no_existe"])
        assert "columna_que_no_existe" not in resultado.columns

    def test_nan_se_preserva(self):
        dataframe = pd.DataFrame({"en_gira_activa": [True, None, False]})
        resultado = encode_boolean_flags(dataframe, ["en_gira_activa"])
        assert resultado["en_gira_activa"].iloc[0] == 1
        assert pd.isna(resultado["en_gira_activa"].iloc[1])
        assert resultado["en_gira_activa"].iloc[2] == 0


class TestComputeWarmupCutoff:
    def test_columna_is_warmup_creada(self):
        dataframe = crear_df_base()
        resultado = compute_warmup_cutoff(dataframe, "artista", "fecha")
        assert "is_warmup" in resultado.columns

    def test_primeros_cinco_annios_marcados(self):
        # ArtistA: min_fecha=2015-03-01, cutoff=2020-03-01
        # 2015-03-01 y 2016-06-15 deben ser warmup, 2020-09-20 NO
        dataframe = crear_df_base()
        resultado = compute_warmup_cutoff(dataframe, "artista", "fecha", annios_warmup=5)
        filas_artista_a = resultado[resultado["artista"] == "ArtistA"]
        assert filas_artista_a[filas_artista_a["fecha"] == "2015-03-01"]["is_warmup"].iloc[0] is True or \
               bool(filas_artista_a[filas_artista_a["fecha"] == "2015-03-01"]["is_warmup"].iloc[0])
        assert not bool(filas_artista_a[filas_artista_a["fecha"] == "2020-09-20"]["is_warmup"].iloc[0])

    def test_artista_con_un_solo_concierto_marcado_warmup(self):
        dataframe = pd.DataFrame({
            "artista": ["SoloArtist"],
            "pais": ["ES"],
            "fecha": ["2023-01-01"],
        })
        resultado = compute_warmup_cutoff(dataframe, "artista", "fecha", annios_warmup=5)
        assert bool(resultado["is_warmup"].iloc[0])

    def test_retorna_dataframe_con_mismas_filas(self):
        dataframe = crear_df_base()
        resultado = compute_warmup_cutoff(dataframe, "artista", "fecha")
        assert len(resultado) == len(dataframe)


# ---------------------------------------------------------------------------
# Tests: split_utils
# ---------------------------------------------------------------------------

class TestTemporalSplitByArtist:
    def test_tamanio_split_aproximado(self):
        dataframe = crear_df_base()
        train_df, test_df = temporal_split_by_artist(dataframe, "artista", "fecha", ratio_train=0.8)
        total = len(train_df) + len(test_df)
        assert total == len(dataframe)
        assert len(train_df) >= len(test_df)

    def test_no_mezcla_artistas_entre_sets(self):
        dataframe = crear_df_base()
        train_df, test_df = temporal_split_by_artist(dataframe, "artista", "fecha")
        artistas_train = set(train_df["artista"])
        artistas_test = set(test_df["artista"])
        # Artistas con un solo concierto post-warmup pueden estar solo en train
        # Los que tienen test no deben tener filas mezcladas (mismo artista en ambos sets está permitido)
        # pero el orden cronológico debe respetarse: max(train) <= min(test) por artista
        for artista in artistas_train & artistas_test:
            max_fecha_train = pd.to_datetime(
                train_df[train_df["artista"] == artista]["fecha"]
            ).max()
            min_fecha_test = pd.to_datetime(
                test_df[test_df["artista"] == artista]["fecha"]
            ).min()
            assert max_fecha_train <= min_fecha_test

    def test_artista_con_un_concierto_va_a_train(self):
        dataframe = pd.DataFrame({
            "artista": ["SoloArtist", "OtroArtist", "OtroArtist"],
            "pais": ["ES", "US", "FR"],
            "fecha": ["2020-01-01", "2019-06-01", "2021-03-15"],
        })
        train_df, test_df = temporal_split_by_artist(dataframe, "artista", "fecha", ratio_train=0.8)
        assert "SoloArtist" in train_df["artista"].values
        assert "SoloArtist" not in test_df["artista"].values

    def test_excluye_filas_is_warmup_si_columna_existe(self):
        dataframe = crear_df_base()
        dataframe["is_warmup"] = False
        dataframe.loc[0, "is_warmup"] = True
        train_df, test_df = temporal_split_by_artist(dataframe, "artista", "fecha")
        total = len(train_df) + len(test_df)
        assert total == len(dataframe) - 1


# ---------------------------------------------------------------------------
# Tests: metrics_utils
# ---------------------------------------------------------------------------

class TestComputeRegressionMetrics:
    def test_retorna_dict_con_claves_requeridas(self):
        valores_reales = np.array([100.0, 200.0, 300.0])
        valores_predichos = np.array([110.0, 190.0, 310.0])
        metricas = compute_regression_metrics(valores_reales, valores_predichos)
        assert set(metricas.keys()) == {"mae", "rmse", "r2", "mape"}

    def test_prediccion_perfecta(self):
        valores = np.array([100.0, 200.0, 300.0])
        metricas = compute_regression_metrics(valores, valores)
        assert metricas["mae"] == pytest.approx(0.0)
        assert metricas["rmse"] == pytest.approx(0.0)
        assert metricas["r2"] == pytest.approx(1.0)
        assert metricas["mape"] == pytest.approx(0.0)

    def test_mape_none_cuando_todos_reales_son_cero(self):
        valores_reales = np.array([0.0, 0.0, 0.0])
        valores_predichos = np.array([1.0, 2.0, 3.0])
        metricas = compute_regression_metrics(valores_reales, valores_predichos)
        assert metricas["mape"] is None

    def test_tipos_de_retorno_son_float(self):
        valores_reales = np.array([10.0, 20.0])
        valores_predichos = np.array([12.0, 18.0])
        metricas = compute_regression_metrics(valores_reales, valores_predichos)
        assert isinstance(metricas["mae"], float)
        assert isinstance(metricas["rmse"], float)
        assert isinstance(metricas["r2"], float)


class TestLogMetricsToFile:
    def test_crea_fichero_json(self):
        metricas = {"mae": 10.5, "rmse": 15.3, "r2": 0.85, "mape": 12.0}
        with tempfile.TemporaryDirectory() as directorio_temp:
            ruta_salida = Path(directorio_temp) / "metrics.json"
            log_metrics_to_file(metricas, ruta_salida)
            assert ruta_salida.exists()

    def test_fichero_json_contiene_metricas_y_timestamp(self):
        metricas = {"mae": 10.5, "rmse": 15.3, "r2": 0.85, "mape": 12.0}
        with tempfile.TemporaryDirectory() as directorio_temp:
            ruta_salida = Path(directorio_temp) / "metrics.json"
            log_metrics_to_file(metricas, ruta_salida)
            with open(ruta_salida, "r") as archivo:
                contenido = json.load(archivo)
            assert "metricas" in contenido
            assert "timestamp_utc" in contenido
            assert contenido["metricas"]["mae"] == 10.5

    def test_crea_directorio_si_no_existe(self):
        metricas = {"mae": 5.0, "rmse": 7.0, "r2": 0.9, "mape": 5.5}
        with tempfile.TemporaryDirectory() as directorio_temp:
            ruta_salida = Path(directorio_temp) / "subdir" / "metrics.json"
            log_metrics_to_file(metricas, ruta_salida)
            assert ruta_salida.exists()


# ---------------------------------------------------------------------------
# Tests: model_io
# ---------------------------------------------------------------------------

class TestSaveAndLoadModel:
    def test_guardado_y_carga_roundtrip(self):
        modelo_mock = MagicMock()
        modelo_mock.predict.return_value = np.array([1.0, 2.0])

        with tempfile.TemporaryDirectory() as directorio_temp:
            ruta_modelo = Path(directorio_temp) / "model.joblib"
            save_model(modelo_mock, ruta_modelo)
            assert ruta_modelo.exists()

            modelo_cargado = load_model(ruta_modelo)
            assert modelo_cargado is not None

    def test_load_model_lanza_error_si_no_existe(self):
        with pytest.raises(FileNotFoundError):
            load_model("/ruta/que/no/existe/model.joblib")


class TestSaveFeatureImportance:
    def test_crea_csv_con_columnas_correctas(self):
        modelo_mock = MagicMock()
        modelo_mock.feature_importances_ = np.array([0.3, 0.5, 0.2])
        nombres_features = ["feature_a", "feature_b", "feature_c"]

        with tempfile.TemporaryDirectory() as directorio_temp:
            ruta_csv = Path(directorio_temp) / "feature_importance.csv"
            save_feature_importance(modelo_mock, nombres_features, ruta_csv)
            assert ruta_csv.exists()

            dataframe_importancia = pd.read_csv(ruta_csv)
            assert "feature" in dataframe_importancia.columns
            assert "importance" in dataframe_importancia.columns

    def test_csv_ordenado_por_importancia_descendente(self):
        modelo_mock = MagicMock()
        modelo_mock.feature_importances_ = np.array([0.1, 0.8, 0.3])
        nombres_features = ["baja", "alta", "media"]

        with tempfile.TemporaryDirectory() as directorio_temp:
            ruta_csv = Path(directorio_temp) / "feature_importance.csv"
            save_feature_importance(modelo_mock, nombres_features, ruta_csv)
            dataframe_importancia = pd.read_csv(ruta_csv)
            assert dataframe_importancia.iloc[0]["feature"] == "alta"
            assert dataframe_importancia.iloc[-1]["feature"] == "baja"
