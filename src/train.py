"""
train.py
--------
Script principal de treinamento — PD2 MLOps (Instituto Infnet).

Executa 4 experimentos rastreados no MLflow:

  1. RF_sem_reducao_baseline  — Random Forest sem redução de dimensionalidade
  2. RF_com_PCA               — Random Forest + PCA (95% de variância explicada)
  3. RF_com_LDA               — Random Forest + LDA (1 componente, classificação binária)
  4. DT_sem_reducao_baseline  — Decision Tree sem redução (baseline interpretável)

Cada run registra parâmetros, métricas (F1, ROC-AUC) e o artefato do modelo.
Ao final, salva o run_id do modelo campeão em models/champion_run_id.txt.

Uso:
    cd /caminho/para/PD\ 2\ MLops
    python src/train.py

Tempo estimado: 40–80 minutos (GridSearchCV com 270 fits por experimento RF).
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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

# Adiciona src/ ao path para importar data_processing
sys.path.insert(0, str(Path(__file__).parent))
from data_processing import build_preprocessor, cap_outliers, load_data


def _to_float64(X):
    """Garante dtype float64 para evitar overflow em PCA/LDA matmul."""
    return np.asarray(X, dtype=np.float64)


float64_step = FunctionTransformer(_to_float64, validate=False)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/pipeline.yaml") -> dict:
    """Carrega configuração central do projeto."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    log.info("Configuração carregada de: %s", config_path)
    return cfg


# ─── Dados ───────────────────────────────────────────────────────────────────

def prepare_data(cfg: dict):
    """
    Carrega dataset, aplica outlier capping e realiza train/test split estratificado.

    Returns
    -------
    X_train, X_test, y_train, y_test, features : tuple
    """
    df = load_data(cfg["data"]["raw_path"])

    for entry in cfg["features"]["outlier_cap"]:
        df = cap_outliers(df, col=entry["col"], upper_pct=entry["upper_pct"])

    target = cfg["data"]["target"]
    features = cfg["features"]["all"]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    log.info(
        "Split estratificado: treino=%d (%.1f%% positivo) | teste=%d (%.1f%% positivo)",
        len(X_train), 100 * y_train.mean(),
        len(X_test), 100 * y_test.mean(),
    )

    return X_train, X_test, y_train, y_test, features


# ─── Núcleo de treinamento ───────────────────────────────────────────────────

def _train_and_log(
    run_name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: dict,
    extra_params: dict = None,
) -> tuple:
    """
    Executa GridSearchCV e registra tudo no MLflow.

    Returns
    -------
    (best_pipeline, run_id, f1_test) : tuple
    """
    cv_cfg = cfg["cross_validation"]
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cfg["data"]["random_state"],
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Parâmetros contextuais (redução, modelo)
        if extra_params:
            mlflow.log_params(extra_params)

        mlflow.log_param("cv_folds", cv_cfg["n_splits"])
        mlflow.log_param("test_size", cfg["data"]["test_size"])
        mlflow.log_param("random_state", cfg["data"]["random_state"])

        # GridSearchCV
        log.info("[%s] Iniciando GridSearchCV (%d folds)...", run_name, cv_cfg["n_splits"])
        t0 = time.perf_counter()
        gs = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            refit=True,
            verbose=0,
            error_score="raise",
        )
        gs.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        best = gs.best_estimator_

        # Inferência e métricas no holdout
        t_inf = time.perf_counter()
        y_pred = best.predict(X_test)
        infer_time = time.perf_counter() - t_inf

        y_proba = best.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        f1_cv = gs.best_score_

        # Log de parâmetros do melhor estimador
        mlflow.log_params(gs.best_params_)

        # Log de métricas
        mlflow.log_metric("f1_score_test", round(f1, 6))
        mlflow.log_metric("roc_auc_test", round(roc, 6))
        mlflow.log_metric("f1_score_cv", round(f1_cv, 6))
        mlflow.log_metric("train_time_sec", round(train_time, 3))
        mlflow.log_metric("infer_time_ms", round(infer_time * 1000, 4))

        # Artefato do modelo
        mlflow.sklearn.log_model(best, artifact_path="model")

        log.info(
            "[%s] CONCLUÍDO | F1=%.4f | ROC-AUC=%.4f | CV F1=%.4f | Treino=%.1fs | run_id=%s",
            run_name, f1, roc, f1_cv, train_time, run_id,
        )

    return best, run_id, f1


# ─── Experimentos ─────────────────────────────────────────────────────────────

def experimento_rf_baseline(X_train, X_test, y_train, y_test, features, cfg):
    """
    Experimento 1: Random Forest sem redução de dimensionalidade.

    Este é o modelo baseline e espera-se que seja o campeão, pois:
    - 10 features não sofrem com maldição da dimensionalidade
    - RF é robusto e não requer dados normalizados (StandardScaler é aplicado
      apenas por consistência com os experimentos que usam PCA/LDA)
    - class_weight="balanced" compensa o desbalanceamento de 6.7% positivo
    """
    rf_cfg = cfg["models"]["random_forest"]

    pipeline = Pipeline([
        ("prep", build_preprocessor(features)),
        ("clf", RandomForestClassifier(
            class_weight=rf_cfg["class_weight"],
            random_state=cfg["data"]["random_state"],
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "clf__n_estimators":     rf_cfg["n_estimators"],
        "clf__max_depth":        rf_cfg["max_depth"],
        "clf__max_features":     rf_cfg["max_features"],
        "clf__min_samples_leaf": rf_cfg["min_samples_leaf"],
    }

    return _train_and_log(
        run_name="RF_sem_reducao_baseline",
        pipeline=pipeline,
        param_grid=param_grid,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        cfg=cfg,
        extra_params={
            "reducao_dimensionalidade": "nenhuma",
            "modelo": "RandomForest",
        },
    )


def experimento_rf_pca(X_train, X_test, y_train, y_test, features, cfg):
    """
    Experimento 2: Random Forest + PCA.

    PCA(n_components=0.95): sklearn seleciona automaticamente o número mínimo de
    componentes que explicam pelo menos 95% da variância total. Com 10 features
    correlacionadas (atrasos de 30-59, 60-89 e 90+ dias são altamente correlacionados),
    esperam-se ~7-8 componentes.

    Limitação esperada: PCA elimina a interpretabilidade direta das features
    originais (ex: RevolvingUtilizationOfUnsecuredLines não é mais identificável).
    Isso é relevante para conformidade com LGPD (Art. 20) e resoluções BACEN.
    """
    rf_cfg = cfg["models"]["random_forest"]
    pca_var = cfg["dimensionality_reduction"]["pca"]["variance_threshold"]

    pipeline = Pipeline([
        ("prep", build_preprocessor(features)),
        ("to_f64", float64_step),
        ("pca", PCA(n_components=pca_var, random_state=cfg["data"]["random_state"])),
        ("clf", RandomForestClassifier(
            class_weight=rf_cfg["class_weight"],
            random_state=cfg["data"]["random_state"],
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "clf__n_estimators":     rf_cfg["n_estimators"],
        "clf__max_depth":        rf_cfg["max_depth"],
        "clf__max_features":     rf_cfg["max_features"],
        "clf__min_samples_leaf": rf_cfg["min_samples_leaf"],
    }

    best, run_id, f1 = _train_and_log(
        run_name="RF_com_PCA",
        pipeline=pipeline,
        param_grid=param_grid,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        cfg=cfg,
        extra_params={
            "reducao_dimensionalidade": "PCA",
            "modelo": "RandomForest",
            "pca_variance_threshold": pca_var,
        },
    )

    # Log retroativo do número real de componentes escolhido pelo PCA
    n_comp = best.named_steps["pca"].n_components_
    log.info("[RF_com_PCA] PCA escolheu %d componentes para %.0f%% de variância", n_comp, pca_var * 100)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("pca_n_components_chosen", n_comp)

    return best, run_id, f1


def experimento_rf_lda(X_train, X_test, y_train, y_test, features, cfg):
    """
    Experimento 3: Random Forest + LDA.

    LDA para classificação binária: n_components = min(n_classes - 1, n_features) = 1.
    Isso é uma LIMITAÇÃO MATEMÁTICA, não uma escolha arbitrária.

    Consequência: todo o espaço de 10 dimensões é projetado em 1 único eixo discriminante.
    Espera-se queda significativa no F1 comparado ao baseline sem redução.

    Vantagem do LDA sobre PCA neste contexto: LDA é supervisionado — maximiza a
    separação entre classes, não a variância total. O único componente resultante é
    a combinação linear das features que melhor separa adimplentes de inadimplentes.

    Nota técnica: sklearn Pipeline propaga y corretamente para steps supervisionados
    como LDA no fit(X, y) — nenhuma workaround é necessária.
    """
    rf_cfg = cfg["models"]["random_forest"]
    lda_n = cfg["dimensionality_reduction"]["lda"]["n_components"]

    pipeline = Pipeline([
        ("prep", build_preprocessor(features)),
        ("to_f64", float64_step),
        ("lda", LDA(n_components=lda_n)),
        ("clf", RandomForestClassifier(
            class_weight=rf_cfg["class_weight"],
            random_state=cfg["data"]["random_state"],
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "clf__n_estimators":     rf_cfg["n_estimators"],
        "clf__max_depth":        rf_cfg["max_depth"],
        "clf__max_features":     rf_cfg["max_features"],
        "clf__min_samples_leaf": rf_cfg["min_samples_leaf"],
    }

    return _train_and_log(
        run_name="RF_com_LDA",
        pipeline=pipeline,
        param_grid=param_grid,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        cfg=cfg,
        extra_params={
            "reducao_dimensionalidade": "LDA",
            "modelo": "RandomForest",
            "lda_n_components": lda_n,
            "lda_type": "LinearDiscriminantAnalysis_supervisionado",
        },
    )


def experimento_dt_baseline(X_train, X_test, y_train, y_test, features, cfg):
    """
    Experimento 4: Decision Tree sem redução (baseline interpretável).

    Serve como comparação para justificar a adoção do Random Forest:
    - Decision Tree é totalmente auditável (regras extraíveis com export_text)
    - Porém sofre de overfitting mesmo com regularização (GridSearchCV mitiga)
    - Comparar com RF demonstra o ganho de F1 do ensemble vs. interpretabilidade
    """
    dt_cfg = cfg["models"]["decision_tree"]

    pipeline = Pipeline([
        ("prep", build_preprocessor(features)),
        ("clf", DecisionTreeClassifier(
            class_weight=dt_cfg["class_weight"],
            random_state=cfg["data"]["random_state"],
        )),
    ])

    param_grid = {
        "clf__criterion":          dt_cfg["criterion"],
        "clf__max_depth":          dt_cfg["max_depth"],
        "clf__min_samples_split":  dt_cfg["min_samples_split"],
        "clf__min_samples_leaf":   dt_cfg["min_samples_leaf"],
    }

    return _train_and_log(
        run_name="DT_sem_reducao_baseline",
        pipeline=pipeline,
        param_grid=param_grid,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        cfg=cfg,
        extra_params={
            "reducao_dimensionalidade": "nenhuma",
            "modelo": "DecisionTree",
        },
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("PD2 MLOps — Iniciando treinamento de 4 experimentos")
    log.info("=" * 60)

    cfg = load_config()

    # Configurar MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    log.info("MLflow configurado: uri=%s | experiment=%s",
             cfg["mlflow"]["tracking_uri"], cfg["mlflow"]["experiment_name"])

    # Preparar dados (uma única vez, compartilhado entre experimentos)
    X_train, X_test, y_train, y_test, features = prepare_data(cfg)

    # ── Executar experimentos sequencialmente ──────────────────────────────
    resultados = []

    log.info("\n=== EXPERIMENTO 1/4: RF sem redução ===")
    best1, run_id1, f1_1 = experimento_rf_baseline(
        X_train, X_test, y_train, y_test, features, cfg)
    resultados.append(("RF_sem_reducao_baseline", run_id1, f1_1))

    log.info("\n=== EXPERIMENTO 2/4: RF + PCA ===")
    best2, run_id2, f1_2 = experimento_rf_pca(
        X_train, X_test, y_train, y_test, features, cfg)
    resultados.append(("RF_com_PCA", run_id2, f1_2))

    log.info("\n=== EXPERIMENTO 3/4: RF + LDA ===")
    best3, run_id3, f1_3 = experimento_rf_lda(
        X_train, X_test, y_train, y_test, features, cfg)
    resultados.append(("RF_com_LDA", run_id3, f1_3))

    log.info("\n=== EXPERIMENTO 4/4: Decision Tree baseline ===")
    best4, run_id4, f1_4 = experimento_dt_baseline(
        X_train, X_test, y_train, y_test, features, cfg)
    resultados.append(("DT_sem_reducao_baseline", run_id4, f1_4))

    # ── Selecionar modelo campeão ──────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("RESUMO DOS EXPERIMENTOS")
    log.info("=" * 60)
    for nome, run_id, f1 in resultados:
        log.info("  %-35s | F1=%.4f | run_id=%s", nome, f1, run_id)

    campeao = max(resultados, key=lambda x: x[2])
    nome_campeao, run_id_campeao, f1_campeao = campeao

    log.info("\nMODELO CAMPEÃO: %s (F1=%.4f)", nome_campeao, f1_campeao)

    # Salvar run_id do campeão para o Streamlit
    Path("models").mkdir(exist_ok=True)
    champion_file = Path("models/champion_run_id.txt")
    champion_file.write_text(run_id_campeao)
    log.info("run_id do campeão salvo em: %s", champion_file)

    log.info("\nPróximos passos:")
    log.info("  1. mlflow ui                     → visualizar experimentos em http://localhost:5000")
    log.info("  2. streamlit run app/app.py      → iniciar interface de inferência")
    log.info("  3. MODEL_URI=runs:/%s/model     → forçar modelo específico", run_id_campeao)


if __name__ == "__main__":
    main()
