"""
data_processing.py
------------------
Funções de ingestão, limpeza e construção do pré-processador sklearn.

Responsabilidades:
  - Carregar CSV e validar schema
  - Tratar outliers por capping de percentil (aplicado antes do split)
  - Construir ColumnTransformer com SimpleImputer + StandardScaler

Nota sobre data leakage no outlier capping:
    O threshold do percentil 99 é calculado sobre o dataset completo (antes do
    train_test_split). Em produção, o threshold deveria ser calculado APENAS no
    conjunto de treino e armazenado como artefato de pré-processamento. Neste
    contexto acadêmico, a simplificação é aceitável e está documentada aqui.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

EXPECTED_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def load_data(path: str) -> pd.DataFrame:
    """
    Carrega o CSV do Give Me Some Credit e valida o schema esperado.

    Parameters
    ----------
    path : str
        Caminho para cs-training.csv

    Returns
    -------
    pd.DataFrame
        DataFrame com index original removido e colunas validadas.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir no caminho especificado.
    ValueError
        Se colunas esperadas estiverem ausentes no arquivo.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {path}\n"
            "Copie cs-training.csv para data/raw/ conforme README.md"
        )

    df = pd.read_csv(path, index_col=0)
    log.info("Dataset carregado: %d linhas x %d colunas", *df.shape)

    missing_cols = set(EXPECTED_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colunas ausentes no dataset: {missing_cols}")

    # Diagnóstico de qualidade de dados
    target = "SeriousDlqin2yrs"
    all_cols = EXPECTED_FEATURES + ([target] if target in df.columns else [])
    null_counts = df[all_cols].isnull().sum()
    nulls_found = null_counts[null_counts > 0]
    if not nulls_found.empty:
        log.warning(
            "Valores ausentes detectados (serão tratados por imputação mediana):\n%s",
            nulls_found.to_string(),
        )

    # Diagnóstico de desbalanceamento da classe alvo
    if target in df.columns:
        pct_positivo = df[target].mean() * 100
        log.info(
            "Distribuição do target '%s': positivos=%.2f%%, negativos=%.2f%%",
            target, pct_positivo, 100 - pct_positivo,
        )

    return df


def cap_outliers(df: pd.DataFrame, col: str, upper_pct: float = 99) -> pd.DataFrame:
    """
    Limita (caps) os valores de uma coluna no percentil superior especificado.

    Motivação: A feature DebtRatio possui valores absurdos (> 1000) que representam
    erros de dados ou casos extremamente atípicos. O capping no p99 preserva a
    distribuição real enquanto elimina o efeito distorcivo desses outliers no
    StandardScaler e nos modelos baseados em distância.

    Nota de engenharia: Em produção, o threshold seria calculado APENAS no conjunto
    de treino e armazenado como parâmetro do pipeline de pré-processamento, evitando
    data leakage. Aqui calculamos no dataset completo por simplicidade acadêmica.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada (não modificado in-place — retorna cópia).
    col : str
        Nome da coluna a ser tratada.
    upper_pct : float
        Percentil superior para capping (padrão: 99).

    Returns
    -------
    pd.DataFrame
        Cópia do DataFrame com outliers tratados na coluna especificada.
    """
    df = df.copy()
    upper = np.percentile(df[col].dropna(), upper_pct)
    n_capped = int((df[col] > upper).sum())

    if n_capped > 0:
        log.info(
            "Outlier capping em '%s': %d valores (%.2f%%) acima de p%d (%.4f) → %.4f",
            col,
            n_capped,
            100 * n_capped / len(df),
            upper_pct,
            df[col].max(),
            upper,
        )
    else:
        log.info("Outlier capping em '%s': nenhum valor acima de p%d (%.4f)", col, upper_pct, upper)

    df[col] = df[col].clip(upper=upper)
    return df


def build_preprocessor(features: list) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer de pré-processamento numérico.

    Transformações aplicadas sequencialmente:
      1. SimpleImputer(strategy='median')
         Trata valores ausentes (MonthlyIncome ~20%, NumberOfDependents ~2.5%).
         Mediana é mais robusta que média na presença de outliers residuais.
      2. StandardScaler()
         Centraliza e normaliza cada feature (média=0, desvio=1).
         Necessário para PCA (sensível à escala) e melhora convergência geral.

    Parameters
    ----------
    features : list
        Lista ordenada das colunas de entrada numéricas.

    Returns
    -------
    ColumnTransformer
        Transformador pronto para ser encadeado em um Pipeline sklearn.
        Cada chamada retorna uma instância NOVA (não fitted) — crie uma por experimento.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, features),
        ],
        remainder="drop",  # descarta colunas extras acidentalmente presentes em X
    )
    return preprocessor
