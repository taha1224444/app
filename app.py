from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

GENDER_VALUES = ["Female", "Male", "Other"]
SMOKING_VALUES = ["No Info", "current", "ever", "former", "never", "not current"]
SMOKING_LABELS = {
    "No Info": "No Info",
    "current": "Current",
    "ever": "Ever",
    "former": "Former",
    "never": "Never",
    "not current": "Not Current",
}

NUMERIC_LIMITS = {
    "age": (0, 120),
    "bmi": (0, 100),
    "HbA1c_level": (0, 20),
    "blood_glucose_level": (0, 500),
}

COLUMN_ALIASES = {
    "gender": "gender",
    "age": "age",
    "hypertension": "hypertension",
    "heart_disease": "heart_disease",
    "smoking_history": "smoking_history",
    "bmi": "bmi",
    "hba1c": "HbA1c_level",
    "hba1c_level": "HbA1c_level",
    "blood_glucose_level": "blood_glucose_level",
    "blood_glucose": "blood_glucose_level",
    "glucose": "blood_glucose_level",
}

MODEL_WARNINGS: list[str] = []


def load_joblib_model(model_name: str, filename: str) -> Any | None:
    try:
        return joblib.load(BASE_DIR / filename)
    except Exception as exc:
        MODEL_WARNINGS.append(f"{model_name} is unavailable: {exc}.")
        return None


def load_catboost_model(filename: str) -> CatBoostClassifier | None:
    try:
        model = CatBoostClassifier()
        model.load_model(str(BASE_DIR / filename))
        return model
    except Exception as exc:
        MODEL_WARNINGS.append(f"CatBoost is unavailable: {exc}.")
        return None


def load_ann_model(filename: str) -> Any | None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import tensorflow as tf  # type: ignore
    except Exception as exc:
        MODEL_WARNINGS.append(
            "ANN is unavailable because TensorFlow could not be imported in this environment."
        )
        app.logger.warning("TensorFlow import failed: %s", exc)
        return None

    try:
        return tf.keras.models.load_model(BASE_DIR / filename)
    except Exception as exc:
        MODEL_WARNINGS.append(f"ANN is unavailable: {exc}.")
        return None


xgb_model = load_joblib_model("XGBoost", "xgb_model.pkl")
cat_model = load_catboost_model("catboost_model.cbm")
scaler = load_joblib_model("Scaler", "scaler.pkl")
ann_model = load_ann_model("ann_pso_model.keras")


def json_error(message: str, status_code: int = 400) -> tuple[Any, int]:
    return jsonify({"error": message}), status_code


def risk_level(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def normalize_header(column_name: str) -> str:
    return column_name.strip().casefold().replace(" ", "_")


def decode_choice(index: int, choices: list[str]) -> str:
    return choices[int(index)]


def coerce_category(value: Any, field_name: str, choices: list[str]) -> int:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"'{field_name}' cannot be empty.")

        try:
            numeric_value = float(text)
        except ValueError:
            normalized_map = {choice.casefold(): index for index, choice in enumerate(choices)}
            key = text.casefold()
            if key not in normalized_map:
                readable = ", ".join(choices)
                raise ValueError(f"'{field_name}' must be one of: {readable}.")
            return normalized_map[key]

        if not numeric_value.is_integer():
            raise ValueError(f"'{field_name}' must be a whole number or valid label.")
        index = int(numeric_value)
    elif isinstance(value, (int, np.integer)):
        index = int(value)
    elif isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise ValueError(f"'{field_name}' must be a whole number or valid label.")
        index = int(value)
    else:
        raise ValueError(f"'{field_name}' has an unsupported value.")

    if not 0 <= index < len(choices):
        raise ValueError(f"'{field_name}' must be between 0 and {len(choices) - 1}.")
    return index


def coerce_binary(value: Any, field_name: str) -> int:
    if isinstance(value, str):
        text = value.strip()
        if text in {"0", "1"}:
            return int(text)
        raise ValueError(f"'{field_name}' must be 0 or 1.")

    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return int(value)

    if isinstance(value, (float, np.floating)) and float(value).is_integer() and int(value) in {0, 1}:
        return int(value)

    raise ValueError(f"'{field_name}' must be 0 or 1.")


def coerce_number(value: Any, field_name: str, minimum: float, maximum: float) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be a valid number.") from exc

    if not np.isfinite(numeric_value):
        raise ValueError(f"'{field_name}' must be finite.")
    if not minimum <= numeric_value <= maximum:
        raise ValueError(f"'{field_name}' must be between {minimum} and {maximum}.")

    return numeric_value


def pick_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise ValueError(f"Missing required field: '{keys[0]}'.")


def build_single_feature_frame(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    record = {
        "gender": coerce_category(pick_value(payload, "gender"), "gender", GENDER_VALUES),
        "age": coerce_number(pick_value(payload, "age"), "age", *NUMERIC_LIMITS["age"]),
        "hypertension": coerce_binary(pick_value(payload, "hypertension"), "hypertension"),
        "heart_disease": coerce_binary(pick_value(payload, "heart_disease"), "heart_disease"),
        "smoking_history": coerce_category(
            pick_value(payload, "smoking_history"),
            "smoking_history",
            SMOKING_VALUES,
        ),
        "bmi": coerce_number(pick_value(payload, "bmi"), "bmi", *NUMERIC_LIMITS["bmi"]),
        "HbA1c_level": coerce_number(
            pick_value(payload, "hba1c", "HbA1c_level"),
            "HbA1c_level",
            *NUMERIC_LIMITS["HbA1c_level"],
        ),
        "blood_glucose_level": coerce_number(
            pick_value(payload, "glucose", "blood_glucose_level"),
            "blood_glucose_level",
            *NUMERIC_LIMITS["blood_glucose_level"],
        ),
    }

    frame = pd.DataFrame([record], columns=FEATURES)
    return frame, record


def build_profile(record: dict[str, Any]) -> dict[str, str]:
    return {
        "Gender": decode_choice(record["gender"], GENDER_VALUES),
        "Age": f"{record['age']:.0f}",
        "Hypertension": "Yes" if record["hypertension"] else "No",
        "Heart Disease": "Yes" if record["heart_disease"] else "No",
        "Smoking": SMOKING_LABELS[decode_choice(record["smoking_history"], SMOKING_VALUES)],
        "BMI": f"{record['bmi']:.1f}",
        "HbA1c": f"{record['HbA1c_level']:.1f}",
        "Glucose": f"{record['blood_glucose_level']:.0f} mg/dL",
    }


def rename_batch_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    seen: dict[str, str] = {}

    for column in dataframe.columns:
        canonical = COLUMN_ALIASES.get(normalize_header(column))
        if not canonical:
            continue
        if canonical in seen:
            raise ValueError(f"CSV contains duplicate versions of '{canonical}'.")
        seen[canonical] = column
        rename_map[column] = canonical

    return dataframe.rename(columns=rename_map)


def invalid_rows_message(mask: pd.Series) -> str:
    rows = [str(int(index) + 2) for index in mask[mask].index[:5]]
    return ", ".join(rows)


def encode_batch_category(dataframe: pd.DataFrame, field_name: str, choices: list[str]) -> pd.Series:
    encoded_values: list[int] = []
    invalid_rows: list[int] = []

    for index, value in dataframe[field_name].items():
        try:
            encoded_values.append(coerce_category(value, field_name, choices))
        except ValueError:
            invalid_rows.append(int(index) + 2)

    if invalid_rows:
        preview = ", ".join(str(row) for row in invalid_rows[:5])
        raise ValueError(f"Column '{field_name}' contains invalid values at CSV rows: {preview}.")

    return pd.Series(encoded_values, index=dataframe.index, dtype=int)


def prepare_batch_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        raise ValueError("The uploaded CSV is empty.")

    prepared = rename_batch_columns(dataframe.copy())

    missing_columns = [column for column in FEATURES if column not in prepared.columns]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing_columns)}.")

    prepared = prepared.loc[:, FEATURES].copy()

    prepared["gender"] = encode_batch_category(prepared, "gender", GENDER_VALUES)
    prepared["smoking_history"] = encode_batch_category(prepared, "smoking_history", SMOKING_VALUES)

    for column in ("hypertension", "heart_disease"):
        numeric_series = pd.to_numeric(prepared[column], errors="coerce")
        invalid_mask = numeric_series.isna() | (numeric_series % 1 != 0) | (~numeric_series.isin([0, 1]))
        if invalid_mask.any():
            raise ValueError(
                f"Column '{column}' must contain only 0 or 1 values. Check CSV rows: {invalid_rows_message(invalid_mask)}."
            )
        prepared[column] = numeric_series.astype(int)

    for column, (minimum, maximum) in NUMERIC_LIMITS.items():
        numeric_series = pd.to_numeric(prepared[column], errors="coerce")
        invalid_mask = numeric_series.isna() | (~np.isfinite(numeric_series)) | (numeric_series < minimum) | (numeric_series > maximum)
        if invalid_mask.any():
            raise ValueError(
                f"Column '{column}' contains invalid values. Check CSV rows: {invalid_rows_message(invalid_mask)}."
            )
        prepared[column] = numeric_series.astype(float)

    return prepared


def predict_ensemble(dataframe: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model_scores: dict[str, np.ndarray] = {}

    if xgb_model is not None:
        model_scores["XGBoost"] = xgb_model.predict_proba(dataframe)[:, 1].astype(float)
    if et_model is not None:
        model_scores["ExtraTrees"] = et_model.predict_proba(dataframe)[:, 1].astype(float)
    if cat_model is not None:
        model_scores["CatBoost"] = cat_model.predict_proba(dataframe)[:, 1].astype(float)
    if ann_model is not None and scaler is not None:
        scaled_frame = pd.DataFrame(
            scaler.transform(dataframe),
            columns=FEATURES,
            index=dataframe.index,
        )
        model_scores["ANN"] = ann_model.predict(scaled_frame, verbose=0).ravel().astype(float)

    if not model_scores:
        raise RuntimeError("No prediction models are currently available.")

    final_risk = np.mean(np.column_stack(list(model_scores.values())), axis=1)
    return final_risk, model_scores


@app.errorhandler(413)
def file_too_large(_: Exception) -> tuple[Any, int]:
    return json_error("The uploaded file is too large. Please keep CSV uploads under 5 MB.", 413)


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        feature_columns=FEATURES,
        gender_choices=[{"value": value, "label": value} for value in GENDER_VALUES],
        smoking_choices=[{"value": value, "label": SMOKING_LABELS[value]} for value in SMOKING_VALUES],
        model_warnings=MODEL_WARNINGS,
    )


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Any, int] | Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return json_error("Request body must be valid JSON.", 400)

    try:
        feature_frame, record = build_single_feature_frame(payload)
        final_risk, model_scores = predict_ensemble(feature_frame)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception:
        app.logger.exception("Single prediction failed.")
        return json_error("Prediction failed due to an internal server error.", 500)

    rounded_models = {
        name: round(float(scores[0]), 3)
        for name, scores in model_scores.items()
    }
    score = float(final_risk[0])

    return jsonify({
        "final_risk": round(score, 3),
        "risk_level": risk_level(score),
        "lead_model": max(rounded_models, key=rounded_models.get),
        "models": rounded_models,
        "model_warnings": MODEL_WARNINGS,
        "features": feature_frame.to_dict(orient="records")[0],
        "profile": build_profile(record),
    })


@app.route("/upload", methods=["POST"])
def upload() -> tuple[Any, int] | Any:
    file = request.files.get("file")
    if file is None or not file.filename:
        return json_error("Please upload a CSV file.", 400)

    if Path(file.filename).suffix.lower() != ".csv":
        return json_error("Only CSV uploads are supported.", 400)

    try:
        dataframe = pd.read_csv(file)
        prepared = prepare_batch_frame(dataframe)
        final_risk, model_scores = predict_ensemble(prepared)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception:
        app.logger.exception("Batch prediction failed.")
        return json_error("Batch prediction failed due to an internal server error.", 500)

    risk_distribution = {
        "low": int((final_risk < 0.4).sum()),
        "medium": int(((final_risk >= 0.4) & (final_risk < 0.7)).sum()),
        "high": int((final_risk >= 0.7).sum()),
    }
    batch_size = int(len(prepared))
    model_averages = {
        name: round(float(scores.mean()), 3)
        for name, scores in model_scores.items()
    }

    return jsonify({
        "batch_size": batch_size,
        "average_risk": round(float(final_risk.mean()), 3),
        "median_risk": round(float(np.median(final_risk)), 3),
        "highest_risk": round(float(final_risk.max()), 3),
        "high_risk_cases": risk_distribution["high"],
        "lead_model": max(model_averages, key=model_averages.get),
        "model_warnings": MODEL_WARNINGS,
        "model_averages": model_averages,
        "risk_distribution": risk_distribution,
        "risk_percentages": {
            key: round(value / batch_size, 3)
            for key, value in risk_distribution.items()
        },
    })


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "9998")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
