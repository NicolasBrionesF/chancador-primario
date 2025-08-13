from typing import Dict, Tuple
import pandas as pd
import numpy as np

def make_train_set(df: pd.DataFrame, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Ejemplo de creación de dataset:
    - Ventanas deslizantes de 'window_records'
    - Agrega std y tendencias si se solicita
    Nota: adapta la selección de columnas objetivo/labels según tu notebook.
    """
    win = cfg.get("window_records", 4)
    use_std = cfg.get("use_std", True)
    use_trend = cfg.get("use_trend", True)

    # Selecciona columnas numéricas (ajusta a tu caso)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X_list = []
    for c in num_cols:
        arr = df[c].values
        # rolling window to last value
        roll = pd.Series(arr).rolling(win, min_periods=win).apply(lambda x: x.iloc[-1])
        X_list.append(roll.values.reshape(-1, 1))

        if use_std:
            stdv = pd.Series(arr).rolling(win, min_periods=win).std()
            X_list.append(stdv.values.reshape(-1, 1))

        if use_trend:
            # diferencia simple
            diff = pd.Series(arr).diff().rolling(win, min_periods=win).mean()
            X_list.append(diff.values.reshape(-1, 1))

    X = np.hstack([x for x in X_list if x is not None])
    # Placeholder de etiquetas (reemplazar por y real de tu cuaderno)
    y = np.zeros(len(df))
    meta = {"num_cols": num_cols, "window": win}
    # elimina filas iniciales con NaN generados por rolling
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    return X, y, meta
