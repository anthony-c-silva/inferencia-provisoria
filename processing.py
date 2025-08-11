"""
processamento.py  –  BioAiLab (FastAPI + MongoDB)

Toda a lógica de cálculo foi portada do código antigo
(AWS Lambda + DynamoDB).  Não há boto3, Decimal para gravar,
nem funções Lambda.  Os dados agora são lidos do MongoDB
através dos helpers em `database.py`.
"""

from __future__ import annotations

import re
import random
from decimal import Decimal           # só p/ isinstance, não grava em BD
from typing import Any, Dict, List, Tuple
from typing import Any, Dict
from database import get_sensor, get_math_model, get_experiment, get_experiment_data
from models import BacteriasFeatures, PredictionResult, ProcessarEntrada
from inference import predict
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --------------------------------------------------------------------------- #
# Mongo helpers – implementados em database.py
# --------------------------------------------------------------------------- #
from database import (
    get_sensor,
    get_math_model,
    get_experiment,
    get_experiment_data,
)

CONVERSION_FACTOR = 2.78e-6  # usado na conversão Raw → Basic Count
# =========================================================================== #
#                       LEITURA NO MONGO
# =========================================================================== #

def process_and_predict(body: ProcessarEntrada) -> PredictionResult:
    """
    Orquestra o fluxo: busca dados, processa, formata e chama a predição.
    """
    # 1. Executa o processamento original para obter as features
    # A sua função original `fit_data_to_model_simplified` retornava (features, analysisId, experimentId)
    # Vamos chamar ela e pegar apenas o primeiro valor (features)
    features, _, _ = fit_data_to_model_simplified(body.model_dump())
    print(f"➡️ Features processadas: {features}")

    # 2. Mapeia os resultados do processamento para o formato esperado pelo /predict
    # Chave da API 1 -> Chave da API 2
    feature_map = {
        "A": "Amplitude",
        "beta_x": "TempoPontoInflexao",
        "beta_y": "PontoInflexao",
        "gamma_x": "TempoPicoPrimeiraDerivada",
        "gamma_y": "PicoPrimeiraDerivada",
        "delta_x": "TempoPicoSegundaDerivada",
        "delta_y": "PicoSegundaDerivada",
    }
    
    # Cria um dicionário com os nomes corretos
    predict_input_data = {feature_map[key]: value for key, value in features.items()}
    
    # Converte o dicionário para o modelo Pydantic
    bacterias_features = BacteriasFeatures(**predict_input_data)
    print(f"🔄 Features formatadas para predição: {bacterias_features}")

    # 3. Chama a função de predição com os dados formatados
    final_prediction = predict(bacterias_features)
    
    # 4. Retorna o resultado final
    return final_prediction

def fetch_sensor(id: str) -> Dict[str, Any]:
    sensor = get_sensor(id)
    if not sensor:
        raise ValueError(f"Sensor {id} não encontrado.")
    return sensor


def fetch_math_model(id: str) -> Dict[str, Any]:
    model = get_math_model(id)
    if not model:
        raise ValueError(f"Modelo matemático {id} não encontrado.")
    return model


def fetch_experiment(id: str) -> Dict[str, Any]:
    exp = get_experiment(id)
    if not exp:
        raise ValueError(f"Experimento {id} não encontrado.")
    return exp


def fetch_experiment_data_items(id: str) -> List[Dict[str, Any]]:
    items = get_experiment_data(id)
    if not items:
        raise ValueError("Nenhum dado bruto encontrado para o experimento.")
    return items


# =========================================================================== #
#                       ENTRY‑POINT USADO PELO FastAPI
# =========================================================================== #
def process_single_model(body: Dict[str, Any]) -> Tuple[Dict[str, float], str, str]:
    """Mantém assinatura esperada pelo main.py"""
    return fit_data_to_model_simplified(body)


# =========================================================================== #
#                       FUNÇÕES DE CÁLCULO
# =========================================================================== #
def fit_data_to_model_simplified(body: Dict[str, Any]):
    experimentId = body["experimentId"]
    sensorId = body["sensorId"]
    math_modelId = body["mathModelId"]
    sensor_channel = body["sensorChannel"]
    sensor_sub_channel = body["sensorSubChannel"]
    analysisId = body["analysisId"]

    # -- Buscas no Mongo ----------------------------------------------------
    sensor = fetch_sensor(sensorId)
    math_model = fetch_math_model(math_modelId)
    experiment = fetch_experiment(experimentId)
    items = fetch_experiment_data_items(experimentId)

    sensor_name = sensor["sensorName"]
    sensor_category = sensor["sensorCategory"]
    sensor_model = sensor["sensorModel"]

    eq = math_model["equation"]["equation"]
    d_eq = math_model["equation"]["firstDerivative"]
    dd_eq = math_model["equation"]["secondDerivative"]

    light_cfg = experiment.get("light_sensor_config", {})

    # -- parsing bruto  -----------------------------------------------------
    x_vals, y_vals = process_experiment_items(
        items, sensor_name, sensor_channel, sensor_sub_channel
    )

    # -- processamento / ajuste  -------------------------------------------
    _, _, _, _, _, _, features = process_sensor_data(
        sensor_category,
        sensor_channel,
        sensor_sub_channel,
        x_vals,
        y_vals,
        eq,
        d_eq,
        dd_eq,
        sensor_model,
        light_cfg,
    )

    return features, analysisId, experimentId


# --------------------------------------------------------------------------- #
#                      PARSE DO EXPERIMENTO
# --------------------------------------------------------------------------- #
def process_experiment_items(
    items: List[Dict[str, Any]],
    sensor_name: str,
    sensor_channel: str,
    sensor_sub_channel: str,
) -> Tuple[List[float], List[Any]]:
    x_values: List[float] = []
    y_values: List[Any] = []
    print(f"\n📦 Sensor: {sensor_name}, Canal: {sensor_channel}, Subcanal: {sensor_sub_channel}")
    print(f"📊 Total de itens recebidos: {len(items)}")
    for itm in items:
        ts = itm.get("timestamp")
        data = itm.get(sensor_name, {})

        if sensor_channel == "Absoluto":
            y_val = data.get(sensor_sub_channel)
        else:
            y_val = data

        if ts is not None and y_val is not None:
            x_values.append(float(ts))
            y_values.append(y_val)

    return x_values, y_values


# --------------------------------------------------------------------------- #
#                        BRANCH POR TIPO DE SENSOR
# --------------------------------------------------------------------------- #
def process_sensor_data(
    sensor_category: str,
    sensor_channel: str,
    sensor_sub_channel: str,
    x_values: List[float],
    y_values: List[Any],
    eq: str,
    d_eq: str,
    dd_eq: str,
    sensor_model: Dict[str, Any],
    light_cfg: Dict[str, Any],
):
    # condutividade e outros analógicos
    if sensor_category == "Condutividade" or sensor_channel == "Absoluto":
        return apply_model_direct(
            y_values,
            eq,
            d_eq,
            dd_eq,
            sensor_sub_channel,
            x_values,
            light_cfg,
        )
    print(y_values)

    # canais de cor
    if sensor_channel in ["RGB", "CMYK", "XYZ", "HSB", "HSV", "LAB"]:
        print(sensor_channel)

        converted: List[Dict[str, float]] = []
        cfg_key = sensor_category.replace("spectral_", "")
        cfg = light_cfg.get(cfg_key, {})

        for spectral in y_values:
            if not isinstance(spectral, dict):
                raise ValueError("Valores espectrais devem ser dict.")
            basic = {
                k: convert_raw_to_basic_count(v, cfg) for k, v in spectral.items()
            }
            converted.append(
                apply_color_conversion(sensor_channel, basic, sensor_model)
            )
        print(converted)
        channel_vals = [d.get(sensor_sub_channel) for d in converted]
        return apply_model_direct(
            channel_vals,
            eq,
            d_eq,
            dd_eq,
            sensor_sub_channel,
            x_values,
            light_cfg,
        )

    raise ValueError(f"Categoria/canal não suportado: {sensor_category}, {sensor_channel}")


# --------------------------------------------------------------------------- #
#                    CONVERSÕES E AJUSTE DE MODELO
# --------------------------------------------------------------------------- #
def convert_raw_to_basic_count(raw: Any, cfg: Dict[str, Any]) -> float:
    gain = calculate_gain(float(cfg.get("gain", 1)))
    atime = float(cfg.get("atime", 1))
    astep = float(cfg.get("astep", 1))
    return float(raw or 0.0) / (gain * atime * astep * CONVERSION_FACTOR)


def calculate_gain(value: float) -> float:
    mapping = {0: 0.5, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32,
               7: 64, 8: 128, 9: 256, 10: 512}
    return mapping.get(int(value), 1.0)


# --------------------------------------------------------------------------- #
def apply_model_direct(
    y_values: List[Any],
    eq: str,
    d_eq: str,
    dd_eq: str,
    channel: str,
    timestamps: List[float],
    _light_cfg: Dict[str, Any],
):
    # extrai só o canal pedido
    vals = []
    for d in y_values:
        v = d.get(channel) if isinstance(d, dict) else d
        if isinstance(v, Decimal):
            v = float(v)
        vals.append(v)

    vals = [v for v in vals if v is not None]
    if not vals:
        raise ValueError("Nenhum dado válido para processar.")

    x = np.asarray(timestamps, dtype=float)
    y = np.asarray(vals, dtype=float)

    x_norm, y_norm, x_min, x_max, y_min, y_max = normalize_data(x, y)
    params_names = extract_param_names(eq)

    best = find_best_fit(x_norm, y_norm, eq, params_names, 1000, 5e-3)
    if not best:
        raise ValueError("Não foi possível ajustar o modelo.")

    params = best["params"]
    w_start, w_end = best["window_start_x"], best["window_end_x"]

    y_fit_n = evaluate_equation(x_norm, *params, equation=eq, param_names=params_names)
    dy_fit_n = evaluate_equation(x_norm, *params, equation=d_eq, param_names=params_names)
    dyy_fit_n = evaluate_equation(x_norm, *params, equation=dd_eq, param_names=params_names)

    x_fit, y_fit = denormalize_data(x_norm, y_fit_n, x_min, x_max, y_min, y_max)
    _, dy_fit = denormalize_data(x_norm, dy_fit_n, x_min, x_max, y_min, y_max)
    _, dyy_fit = denormalize_data(x_norm, dyy_fit_n, x_min, x_max, y_min, y_max)
    start_x_fit, _ = denormalize_data(w_start, y_fit_n, x_min, x_max, y_min, y_max)
    end_x_fit, _ = denormalize_data(w_end, y_fit_n, x_min, x_max, y_min, y_max)

    x_fit = remove_nan_and_inf(x_fit)
    y_fit = remove_nan_and_inf(y_fit)
    dy_fit = remove_nan_and_inf(dy_fit)
    dyy_fit = remove_nan_and_inf(dyy_fit)

    features = extract_features(x_fit, y_fit, dy_fit, dyy_fit)
    return y_values, y_fit, dy_fit, dyy_fit, start_x_fit, end_x_fit, features


# --------------------------------------------------------------------------- #
#                           UTILS MATEMÁTICOS
# --------------------------------------------------------------------------- #
def extract_param_names(equation: str) -> List[str]:
    return sorted(set(re.findall(r"[A-Z]", equation)))


def normalize_data(x: np.ndarray, y: np.ndarray):

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    print(f"🔍 Normalizando dados:")
    print(f"   x: min={x_min}, max={x_max}")
    print(f"   y: min={y_min}, max={y_max}")
    return (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min), x_min, x_max, y_min, y_max


def denormalize_data(x_n, y_n, x_min, x_max, y_min, y_max):
    x = x_n * (x_max - x_min) + x_min
    y = y_n * (y_max - y_min) + y_min
    return x, y


def evaluate_equation(x, *params, equation: str, param_names: List[str]):
    local = {"x": np.asarray(x)}
    local.update({n: p for n, p in zip(param_names, params)})
    safe = {"np": np, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "pow": np.power}
    try:
        return eval(equation, {"__builtins__": {}}, {**safe, **local})
    except (ZeroDivisionError, FloatingPointError):
        return np.nan


def remove_nan_and_inf(arr):
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), arr[~bad])
    return arr


# --------------------------------------------------------------------------- #
#                       EXTRAÇÃO DE FEATURES
# --------------------------------------------------------------------------- #
def extract_features(x_fit, y_fit, dy_fit, dyy_fit):
    A = y_fit[-1] - y_fit[0]
    idx = np.argmax(dy_fit) if A > 0 else np.argmin(dy_fit)

    beta_x, beta_y = x_fit[idx], y_fit[idx]
    gamma_x, gamma_y = x_fit[idx], dy_fit[idx]

    peaks, _ = find_peaks(dyy_fit if A > 0 else -dyy_fit)
    if peaks.size:
        delta_x, delta_y = x_fit[peaks[0]], dyy_fit[peaks[0]]
    else:
        delta_x, delta_y = x_fit[0], dyy_fit[0]

    # segundos → minutos (relativo ao início)
    beta_x = (beta_x - x_fit[0]) / 60.0
    gamma_x = (gamma_x - x_fit[0]) / 60.0
    delta_x = (delta_x - x_fit[0]) / 60.0

    return {
        "A": A,
        "beta_x": beta_x,
        "beta_y": beta_y,
        "gamma_x": gamma_x,
        "gamma_y": gamma_y,
        "delta_x": delta_x,
        "delta_y": delta_y,
    }


# --------------------------------------------------------------------------- #
#                     DETERMINAÇÃO DA JANELA DE INTERESSE
# --------------------------------------------------------------------------- #
def determine_window(
    x_values,
    y_values,
    ignore_points=30,
    smooth_sigma=10.0,
    min_window_size=10,
    derivative_threshold_factor=0.01,
):
    if len(y_values) < (5 + ignore_points):
        return None, None

    x = np.array(x_values[ignore_points:])
    y = gaussian_filter1d(y_values[ignore_points:], sigma=smooth_sigma)
    dy = np.gradient(y, x)

    decreasing = dy.mean() < 0
    thr = (dy.min() if decreasing else dy.max()) * 0.001
    start_idx = np.where(dy < thr)[0] if decreasing else np.where(dy > thr)[0]
    start = start_idx[0] if start_idx.size else 0

    peak_idx = dy.argmin() if decreasing else dy.argmax()
    stab_thr = derivative_threshold_factor * np.abs(dy).max()
    post = np.where(np.abs(dy[peak_idx:]) < stab_thr)[0]
    end = peak_idx + post[0] if post.size else len(dy) - 1

    if end - start < min_window_size:
        end = min(start + min_window_size, len(dy) - 1)

    return x[start], x[end]


# --------------------------------------------------------------------------- #
#                     AJUSTE COM VÁRIAS TENTATIVAS
# --------------------------------------------------------------------------- #
def find_best_fit(
    x, y, equation, param_names, max_attempts=1000, tolerance=1e-3
):

    w_start, w_end = determine_window(x, y)

    if w_start is None:
        return None

    mask = (x >= w_start) & (x <= w_end)
    x_s, y_s = x[mask], y[mask]
    if len(x_s) < 2:
        return None

    best_err = np.inf
    best_params = None
    bounds = ([-np.inf] * len(param_names), [np.inf] * len(param_names))
    print(f"🧪 Janela estimada: {w_start} → {w_end}")
    print(f"📊 Amostra x_s (len={len(x_s)}): {x_s[:5]}")
    print(f"📊 Amostra y_s (len={len(y_s)}): {y_s[:5]}")

    def f(xv, *p):
        return evaluate_equation(xv, *p, equation=equation, param_names=param_names)

    for _ in range(max_attempts):
        p0 = [random.uniform(-1, 1) for _ in param_names]
        try:
            params, _ = curve_fit(f, x_s, y_s, p0=p0, bounds=bounds)
            mse = ((y_s - f(x_s, *params)) ** 2).mean()
            if mse < best_err:
                best_err, best_params = mse, params
                if mse <= tolerance:
                    break
        except Exception:
            continue

    if best_params is None:
        return None
    return {
        "params": best_params,
        "error": best_err,
        "window_start_x": w_start,
        "window_end_x": w_end,
    }


# --------------------------------------------------------------------------- #
#                     CONVERSÕES DE COR E LAB
# --------------------------------------------------------------------------- #
def apply_color_conversion(conv_type: str, spectral: Dict[str, float], sensor_model):
    ref_matrix = sensor_model.get("conversionMatrix")
    if not ref_matrix:
        raise ValueError("conversionMatrix ausente no sensor_model")
    xyz = spectrum_to_XYZ(spectral, ref_matrix)
    print(xyz)

    if conv_type == "XYZ":
        return {"X": xyz["x_norm"], "Y": xyz["y_norm"], "Z": xyz["z_norm"]}
    if conv_type == "RGB":
        return xyz_to_RGB([xyz["x_norm"], xyz["y_norm"], xyz["z_norm"]])
    if conv_type == "CMYK":
        return rgb_to_CMYK(
            xyz_to_RGB([xyz["x_norm"], xyz["y_norm"], xyz["z_norm"]])
        )
    if conv_type in ("HSB", "HSV"):
        return rgb_to_HSB_HSV(
            xyz_to_RGB([xyz["x_norm"], xyz["y_norm"], xyz["z_norm"]])
        )
    if conv_type == "LAB":
        return xyz_to_LAB([xyz["X_abs"], xyz["Y_abs"], xyz["Z_abs"]])

    raise ValueError(f"Conversão não suportada: {conv_type}")

def spectrum_to_XYZ(spectrum: Dict[str, float], ref_matrix, scale_factor=1):
    """
    Converte o espectro (aceitando chaves maiúsculas e minúsculas) para valores XYZ.
    """
    # funçãp auxiliar para buscar chave ignorando case
    def get_spec(key: str) -> float:
        return float(spectrum.get(key, spectrum.get(key.lower(), 0.0)))

    # monta vetor de 10 elementos: F1–F8, CLR e NIR
    spec = np.array(
        [get_spec(f"F{i}") for i in range(1, 9)]
        + [get_spec("CLR"), get_spec("NIR")],
        dtype=float,
    ) / scale_factor

    ref = np.asarray(ref_matrix, dtype=float)
    # se a matriz estiver transposta (10×3 em vez de 3×10), corrige
    if spec.size != ref.shape[1]:
        if spec.size == ref.shape[0]:
            ref = ref.T
        else:
            raise ValueError(
                f"Dimensão do espectro ({spec.size}) não corresponde à matriz de referência {ref.shape}"
            )

    # cálculo XYZ
    xyz_abs = ref @ spec
    total = float(np.sum(xyz_abs))
    xyz_norm = xyz_abs / total if total > 0 else np.zeros_like(xyz_abs)

    return {
        "X_abs": float(xyz_abs[0]),
        "Y_abs": float(xyz_abs[1]),
        "Z_abs": float(xyz_abs[2]),
        "x_norm": float(xyz_norm[0]),
        "y_norm": float(xyz_norm[1]),
        "z_norm": float(xyz_norm[2]),
    }



def xyz_to_RGB(xyz):
    X, Y, Z = xyz
    r_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return {"R": correct_gamma(r_lin), "G": correct_gamma(g_lin), "B": correct_gamma(b_lin)}


def correct_gamma(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def rgb_to_CMYK(rgb):
    R, G, B = rgb["R"], rgb["G"], rgb["B"]
    K = 1 - max(R, G, B)
    if K < 1:
        C = (1 - R - K) / (1 - K)
        M = (1 - G - K) / (1 - K)
        Y = (1 - B - K) / (1 - K)
    else:
        C = M = Y = 0.0
    return {"C": C, "M": M, "Y": Y, "K": K}


def rgb_to_HSB_HSV(rgb):
    R, G, B = rgb["R"], rgb["G"], rgb["B"]
    mx, mn = max(R, G, B), min(R, G, B)
    delta = mx - mn

    H = 0.0
    if delta:
        if mx == R:
            H = ((G - B) / delta) % 6
        elif mx == G:
            H = ((B - R) / delta) + 2
        else:
            H = ((R - G) / delta) + 4
        H *= 60

    S = 0 if mx == 0 else delta / mx
    return {"H": H, "S": S, "V": mx}


def xyz_to_LAB(xyz_abs):
    X, Y, Z = xyz_abs
    Xn, Yn, Zn = 95.047, 100.0, 108.883
    fx, fy, fz = (lab_f(v / ref) for v, ref in zip((X, Y, Z), (Xn, Yn, Zn)))
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return {"L*": L, "a*": a, "b*": b}


def lab_f(t):
    delta = 6 / 29
    return t ** (1 / 3) if t > delta ** 3 else (t / (3 * delta ** 2)) + 4 / 29
