# Predictive Maintenance for Primary Crusher (Chancador)

**Objetivo:** Detectar fallas con anticipación (15–60 min) en un chancador minero usando series de tiempo (cada 15 s).  
**Enfoque:** Modelos ML (LSTM con TensorFlow/Keras, Random Forest y XGBoost) + análisis riguroso de operación normal y umbrales.  
**Stack:** Python, pandas, scikit-learn, TensorFlow/Keras, XGBoost, matplotlib/seaborn.

## Datos y filtros operacionales
- `RUN = 1` (CNN-3200-CR_0001_MO.RUN)
- `WIC32149.PV > 200` (material presente)
- Omitir 8 registros (≈2 min) tras transición RUN 0→1 (ruido de arranque)
- Frecuencia: 15 s

## Cómo reproducir
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Entrenar
python scripts/train.py --config configs/experiment_baseline.yaml

# Evaluar
python scripts/evaluate.py --config configs/experiment_baseline.yaml
```

> Los notebooks en `notebooks/` contienen EDA y comparativas de modelos. La lógica reusable vive en `src/crusher_predictor/`.

## Resultados (placeholders)
- AUC-PR: _por completar_
- Anticipación media: _por completar_
- Falsas alarmas/día: _por completar_

## Roadmap
- MLflow tracking (opcional)
- Export ONNX/TensorFlow SavedModel
- Pipeline de inferencia (Azure/Databricks)
