from typing import Dict
import json

def full_report(model, cfg: Dict) -> dict:
    # TODO: Implementar evaluación real con tus métricas del notebook
    return {
        "metrics": {
            "auc_pr": None,
            "false_alarms_per_day": None,
            "avg_anticipation_min": None
        },
        "notes": "Completar con evaluación real"
    }

def save_report(report: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
