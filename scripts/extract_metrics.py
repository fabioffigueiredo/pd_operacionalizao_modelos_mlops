"""
Script para extrair precision, recall, confusion matrix e PCA loadings
dos modelos já salvos no MLflow — sem necessidade de retreinar.
"""
import sys
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score
)

# ── paths ──────────────────────────────────────────────────────────────────
BASE = "/Users/fabiofigueiredo/Documents/projetos/PD 2 MLops"
sys.path.insert(0, os.path.join(BASE, "src"))

from data_processing import load_data, cap_outliers, build_preprocessor

mlflow.set_tracking_uri(f"file://{BASE}/mlruns")

# ── carregar config ─────────────────────────────────────────────────────────
with open(f"{BASE}/config/pipeline.yaml") as f:
    cfg = yaml.safe_load(f)

data_cfg   = cfg["data"]
feat_cfg   = cfg["features"]
caps_cfg   = cfg.get("outlier_cap", [])
features   = feat_cfg["all"]
target     = data_cfg["target"]

# ── preparar dados de teste ────────────────────────────────────────────────
print("Carregando dados...")
df = load_data(f"{BASE}/{data_cfg['raw_path']}")

for cap in caps_cfg:
    df = cap_outliers(df, cap["col"], cap["upper_pct"])

from sklearn.model_selection import train_test_split
X = df[features]
y = df[target]
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=data_cfg["test_size"],
    random_state=data_cfg["random_state"],
    stratify=y
)
print(f"X_test shape: {X_test.shape}, positivos: {y_test.sum()} ({y_test.mean():.1%})")

# ── run IDs dos 4 modelos finais ───────────────────────────────────────────
RUNS = {
    "RF_sem_reducao_baseline": "b34f7524b71d40e8870c638c48ef89ce",
    "RF_com_PCA":              "be47f0a236be4692bb03ff45546c05d1",
    "RF_com_LDA":              "e3e1d9bf581b43e2b93d95f7ebe7570c",
    "DT_sem_reducao_baseline": "f123ac7a8d6d473d80a302cc21904a3f",
}

results = {}
pca_loadings = None

for name, run_id in RUNS.items():
    print(f"\n{'='*60}")
    print(f"Modelo: {name}  (run_id={run_id[:8]}...)")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    cm    = confusion_matrix(y_test, y_pred)
    rep   = classification_report(y_test, y_pred, target_names=["Adimplente","Inadimplente"])

    results[name] = {
        "run_id":    run_id,
        "precision": prec,
        "recall":    rec,
        "cm":        cm,
    }

    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{rep}")

    # ── PCA loadings para RF_com_PCA ──────────────────────────────────────
    if name == "RF_com_PCA":
        try:
            # O pipeline tem steps: preprocessor → to_float64 → pca → classifier
            steps = {s: v for s, v in model.steps}
            pca_step = steps.get("pca")
            if pca_step is None:
                # tentar achar PCA dentro de qualquer step
                for sname, sv in model.steps:
                    if hasattr(sv, "components_"):
                        pca_step = sv
                        break

            if pca_step is not None:
                comps = pca_step.components_          # shape (n_components, n_features)
                exp_var = pca_step.explained_variance_ratio_
                loading_df = pd.DataFrame(
                    comps,
                    columns=features,
                    index=[f"PC{i+1}" for i in range(comps.shape[0])]
                )
                pca_loadings = loading_df
                print("\nPCA Loadings (primeiros 3 componentes):")
                print(loading_df.head(3).round(3).to_string())
                print("\nVariância explicada por componente:")
                for i, v in enumerate(exp_var):
                    print(f"  PC{i+1}: {v:.3f} ({v:.1%})")
            else:
                print("PCA step não encontrado no pipeline.")
        except Exception as ex:
            print(f"Erro ao extrair PCA loadings: {ex}")

# ── salvar resultados em CSV para referência ───────────────────────────────
rows = []
for name, r in results.items():
    cm = r["cm"]
    rows.append({
        "modelo":    name,
        "precision": round(r["precision"], 4),
        "recall":    round(r["recall"], 4),
        "TN": cm[0,0], "FP": cm[0,1],
        "FN": cm[1,0], "TP": cm[1,1],
    })
metrics_df = pd.DataFrame(rows)
out = f"{BASE}/reports/metrics_extended.csv"
metrics_df.to_csv(out, index=False)
print(f"\n\nCSV salvo em: {out}")
print(metrics_df.to_string(index=False))

if pca_loadings is not None:
    pca_out = f"{BASE}/reports/pca_loadings.csv"
    pca_loadings.to_csv(pca_out)
    print(f"\nPCA loadings salvos em: {pca_out}")
