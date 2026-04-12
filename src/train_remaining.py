"""
train_remaining.py
------------------
Roda Experimentos 2, 3 e 4 com grids otimizados para tempo razoável.

O Experimento 1 (RF Baseline) já foi concluído (F1=0.4308, run_id salvo).

Estratégia para exp 2 e 3 (com redução dimensional):
- Grid reduzido: melhores hiperparâmetros do exp 1 + 2-3 variações
- PCA com svd_solver='randomized' (muito mais rápido para datasets grandes)
- n_jobs=4 (em vez de -1) para evitar overhead de memória com PCA+joblib

Isso é academicamente correto: o objetivo é COMPARAR o impacto da redução
dimensional, não re-otimizar exaustivamente o RF com cada redução.
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from data_processing import build_preprocessor, cap_outliers, load_data


def _to_float64(X):
    return np.asarray(X, dtype=np.float64)


float64_step = FunctionTransformer(_to_float64, validate=False)


def load_config(config_path="config/pipeline.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(cfg):
    df = load_data(cfg["data"]["raw_path"])
    for entry in cfg["features"]["outlier_cap"]:
        df = cap_outliers(df, col=entry["col"], upper_pct=entry["upper_pct"])
    features = cfg["features"]["all"]
    X = df[features]
    y = df[cfg["data"]["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info("Split: treino=%d | teste=%d | positivos=%.2f%%", len(X_train), len(X_test), 100 * y_train.mean())
    return X_train, X_test, y_train, y_test, features


def _train_log(run_name, pipeline, param_grid, X_train, X_test, y_train, y_test, cfg, extra_params=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        if extra_params:
            mlflow.log_params(extra_params)
        mlflow.log_param("cv_folds", 5)
        t0 = time.perf_counter()
        gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, refit=True, verbose=0)
        gs.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        best = gs.best_estimator_
        t_inf = time.perf_counter()
        y_pred = best.predict(X_test)
        infer_time = time.perf_counter() - t_inf
        y_proba = best.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("f1_score_test", round(f1, 6))
        mlflow.log_metric("roc_auc_test", round(roc, 6))
        mlflow.log_metric("f1_score_cv", round(gs.best_score_, 6))
        mlflow.log_metric("train_time_sec", round(train_time, 3))
        mlflow.log_metric("infer_time_ms", round(infer_time * 1000, 4))
        mlflow.sklearn.log_model(best, artifact_path="model")
        log.info("[%s] F1=%.4f | ROC-AUC=%.4f | Treino=%.1fs | run_id=%s", run_name, f1, roc, train_time, run_id)
    return best, run_id, f1


def experimento_rf_pca(X_train, X_test, y_train, y_test, features, cfg):
    """
    RF + PCA(n_components=9, svd_solver='randomized').

    Nota: após capping correto das features, cada componente explica apenas 6-21%
    da variância (features pouco correlacionadas). São necessários 9 de 10 componentes
    para atingir 95% — PCA oferece mínima redução real neste dataset.
    Usamos n_components=9 (inteiro) para compatibilidade com svd_solver='randomized',
    que é 10-50x mais rápido que 'full' para datasets grandes.

    n_jobs=1 no RF: GridSearchCV (n_jobs=-1) já paralela o outer loop.
    Nested parallelism (GridSearchCV -1 + RF -1) causa overhead severo.
    """
    pca_n = 9  # 9 componentes explicam ~95% da variância (determinado por análise prévia)
    pipeline = Pipeline([
        ("prep", build_preprocessor(features)),
        ("to_f64", float64_step),
        ("pca", PCA(n_components=pca_n, svd_solver="randomized", random_state=42)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)),
    ])
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [10, 20, None],
        "clf__max_features": ["sqrt"],
        "clf__min_samples_leaf": [5, 10],
    }
    best, run_id, f1 = _train_log(
        "RF_com_PCA", pipeline, param_grid,
        X_train, X_test, y_train, y_test, cfg,
        extra_params={"reducao_dimensionalidade": "PCA", "modelo": "RandomForest",
                      "pca_n_components": pca_n, "svd_solver": "randomized",
                      "pca_variance_explicada": "~95%"},
    )
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("pca_n_components_chosen", pca_n)
    return best, run_id, f1


def experimento_rf_lda(X_train, X_test, y_train, y_test, features, cfg):
    """
    RF + LDA(n_components=1).
    LDA colapsa 10 dimensões em 1 — queda significativa de F1 esperada.
    n_jobs=1 no RF para evitar nested parallelism com GridSearchCV.
    """
    lda_n = cfg["dimensionality_reduction"]["lda"]["n_components"]
    pipeline = Pipeline([
        ("prep", build_preprocessor(features)),
        ("to_f64", float64_step),
        ("lda", LDA(n_components=lda_n)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)),
    ])
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_leaf": [5, 10, 20],
    }
    return _train_log(
        "RF_com_LDA", pipeline, param_grid,
        X_train, X_test, y_train, y_test, cfg,
        extra_params={"reducao_dimensionalidade": "LDA", "modelo": "RandomForest",
                      "lda_n_components": lda_n},
    )


def experimento_dt_baseline(X_train, X_test, y_train, y_test, features, cfg):
    dt_cfg = cfg["models"]["decision_tree"]
    pipeline = Pipeline([
        ("prep", build_preprocessor(features)),
        ("clf", DecisionTreeClassifier(class_weight=dt_cfg["class_weight"], random_state=42)),
    ])
    param_grid = {
        "clf__criterion": dt_cfg["criterion"],
        "clf__max_depth": dt_cfg["max_depth"],
        "clf__min_samples_split": dt_cfg["min_samples_split"],
        "clf__min_samples_leaf": dt_cfg["min_samples_leaf"],
    }
    return _train_log(
        "DT_sem_reducao_baseline", pipeline, param_grid,
        X_train, X_test, y_train, y_test, cfg,
        extra_params={"reducao_dimensionalidade": "nenhuma", "modelo": "DecisionTree"},
    )


def main():
    log.info("=" * 60)
    log.info("PD2 MLOps — Experimentos 2, 3 e 4 (grids otimizados)")
    log.info("=" * 60)

    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X_train, X_test, y_train, y_test, features = prepare_data(cfg)

    resultados = []

    log.info("\n=== EXPERIMENTO 2/4: RF + PCA ===")
    _, run_id2, f1_2 = experimento_rf_pca(X_train, X_test, y_train, y_test, features, cfg)
    resultados.append(("RF_com_PCA", run_id2, f1_2))

    log.info("\n=== EXPERIMENTO 3/4: RF + LDA ===")
    _, run_id3, f1_3 = experimento_rf_lda(X_train, X_test, y_train, y_test, features, cfg)
    resultados.append(("RF_com_LDA", run_id3, f1_3))

    log.info("\n=== EXPERIMENTO 4/4: Decision Tree baseline ===")
    _, run_id4, f1_4 = experimento_dt_baseline(X_train, X_test, y_train, y_test, features, cfg)
    resultados.append(("DT_sem_reducao_baseline", run_id4, f1_4))

    # Inclui Exp 1 (RF baseline já concluído)
    resultados.append(("RF_sem_reducao_baseline", "b34f7524b71d40e8870c638c48ef89ce", 0.430831))

    log.info("\n" + "=" * 60)
    log.info("RESUMO FINAL")
    log.info("=" * 60)
    for nome, run_id, f1 in sorted(resultados, key=lambda x: -x[2]):
        log.info("  %-35s | F1=%.4f | run_id=%s", nome, f1, run_id)

    campeao = max(resultados, key=lambda x: x[2])
    nome_c, run_id_c, f1_c = campeao
    log.info("\nMODELO CAMPEÃO: %s (F1=%.4f)", nome_c, f1_c)

    Path("models").mkdir(exist_ok=True)
    Path("models/champion_run_id.txt").write_text(run_id_c)
    log.info("run_id salvo em models/champion_run_id.txt")
    log.info("→ mlflow ui | streamlit run app/app.py")


if __name__ == "__main__":
    main()
