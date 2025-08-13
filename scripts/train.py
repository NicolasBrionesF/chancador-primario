import argparse, yaml, pandas as pd
from src.crusher_predictor import data_prep, features, models

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    # Carga de datos: reemplaza por tu ruta/lectura real
    # df = pd.read_csv('data/tu_archivo.csv', parse_dates=['timestamp'], index_col='timestamp')
    df = pd.DataFrame()  # placeholder: alimentar desde notebook o CSV local

    df = data_prep.load_and_filter(cfg['data'], df)
    X_train, y_train, meta = features.make_train_set(df, cfg['features'])
    model = models.build(cfg['model'])

    # Entrenamiento (RF/XGB) vs LSTM
    if cfg['model']['type'] in ('rf', 'xgboost'):
        model.fit(X_train, y_train)
    else:
        # Ajusta shapes para LSTM (samples, timesteps, features)
        import numpy as np
        X_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        model.fit(X_seq, y_train, epochs=cfg['model']['lstm']['epochs'], batch_size=cfg['model']['lstm']['batch_size'], verbose=1)

    models.save(model, 'models/model.pkl', meta=meta)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
