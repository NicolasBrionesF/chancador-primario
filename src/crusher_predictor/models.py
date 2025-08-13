from typing import Dict, Tuple, Any
import joblib

def build(cfg: Dict):
    t = cfg.get("type", "lstm")
    if t == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=cfg.get("rf", {}).get("n_estimators", 300),
            max_depth=cfg.get("rf", {}).get("max_depth", None),
            n_jobs=-1,
            random_state=42
        )
    if t == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=cfg.get("xgboost", {}).get("n_estimators", 300),
            max_depth=cfg.get("xgboost", {}).get("max_depth", 6),
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
    if t == "lstm":
        # Ejemplo mínimo de Keras
        import tensorflow as tf
        from tensorflow.keras import layers, models
        model = models.Sequential([
            layers.Input(shape=(None, 1)),  # ajusta dimensiones según tu tensor
            layers.LSTM(cfg.get("lstm", {}).get("hidden_size", 64), return_sequences=True),
            layers.Dropout(cfg.get("lstm", {}).get("dropout", 0.2)),
            layers.LSTM(cfg.get("lstm", {}).get("hidden_size", 64)),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC", "Precision", "Recall"])
        return model
    raise ValueError(f"Modelo no soportado: {t}")

def save(model: Any, path: str, meta: dict = None):
    if hasattr(model, 'save'):  # Keras
        model.save(path)
    else:
        joblib.dump({"model": model, "meta": meta}, path)

def load(path: str) -> Tuple[Any, dict]:
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        return model, {}
    except Exception:
        bundle = joblib.load(path)
        return bundle["model"], bundle.get("meta", {})
