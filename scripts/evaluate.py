import argparse, yaml
from src.crusher_predictor import models, evaluation

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    model, meta = models.load('models/model.pkl')
    report = evaluation.full_report(model, cfg)
    evaluation.save_report(report, 'reports/eval_report.json')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
