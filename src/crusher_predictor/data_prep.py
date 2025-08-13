from typing import Dict
import pandas as pd

def load_and_filter(cfg: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros operacionales al dataframe ya cargado.
    - RUN = 1
    - WIC32149.PV > threshold
    - Omitir 8 registros tras transición RUN 0→1
    """
    run_col = cfg.get("run_sensor", "CNN-3200-CR_0001_MO.RUN")
    w_col = cfg.get("weight_sensor", "WIC32149.PV")
    thr = cfg.get("weight_threshold", 200)
    warm = cfg.get("warmup_records_after_run", 8)

    df = df.copy()
    df = df[df[run_col] == 1]
    df = df[df[w_col] > thr]

    # Omitir warmup posterior a 0->1
    run = df[run_col].fillna(0).astype(int)
    starts = (run.diff() == 1).astype(int)
    skip_idx = starts.index[starts == 1]
    to_drop = set()
    for idx in skip_idx:
        # recolecta las siguientes 'warm' filas
        try:
            pos = df.index.get_loc(idx)
            to_drop.update(df.index[pos:pos+warm])
        except Exception:
            pass

    if to_drop:
        df = df.drop(index=[i for i in to_drop if i in df.index])

    return df.sort_index()
