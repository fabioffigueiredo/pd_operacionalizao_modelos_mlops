# PD2 MLOps — Operacionalização de Modelos com MLflow e Streamlit

**Instituto Infnet | Pós-Graduação em Machine Learning, Deep Learning e IA**
**Disciplina:** Operacionalização de Modelos com MLOps
**Aluno:** Fabio Ferreira Figueiredo
**Baseado em:** [PD1 — Fundamentos de ML com Scikit-Learn](https://github.com/fabioffigueiredo/pd-ml-scikit-learning)

---

## Problema de Negócio

Avaliar o risco de crédito de solicitantes prevendo **inadimplência severa** (`SeriousDlqin2yrs`), usando o dataset [Give Me Some Credit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) — 150.000 registros com classe positiva de ~6.7%.

---

## Estrutura do Projeto

```
PD 2 MLops/
├── config/
│   └── pipeline.yaml           # Hiperparâmetros e paths centralizados
├── src/
│   ├── data_processing.py      # Ingestão, outlier capping, preprocessor
│   └── train.py                # 4 experimentos MLflow
├── app/
│   └── app.py                  # Interface Streamlit
├── data/
│   └── raw/
│       └── cs-training.csv     # Dataset (copiar manualmente)
├── models/
│   └── champion_run_id.txt     # Gerado pelo train.py
├── mlruns/                     # Gerado pelo MLflow
├── reports/
│   └── relatorio_tecnico.md   # Relatório final
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

- Python 3.11+
- Dataset `cs-training.csv` do Kaggle (Give Me Some Credit)

---

## Setup

```bash
# 1. Criar e ativar ambiente virtual
python3.11 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Copiar dataset para o local esperado
mkdir -p data/raw
cp /caminho/para/cs-training.csv data/raw/
# Ou, se veio do PD1:
cp pd-ml-scikit-learning-main/archive/cs-training.csv data/raw/
```

---

## Executar Treinamento (4 Experimentos MLflow)

```bash
# Rodar do diretório raiz do projeto
python src/train.py
```

O script executará os seguintes experimentos em sequência:

| Experimento | Modelo | Redução Dimensional |
|-------------|--------|---------------------|
| `RF_sem_reducao_baseline` | Random Forest | Nenhuma |
| `RF_com_PCA` | Random Forest | PCA (95% variância) |
| `RF_com_LDA` | Random Forest | LDA (1 componente) |
| `DT_sem_reducao_baseline` | Decision Tree | Nenhuma |

**Tempo estimado:** 40–80 minutos (GridSearchCV com validação cruzada 5-fold).

Ao concluir, o `run_id` do modelo campeão é salvo automaticamente em `models/champion_run_id.txt`.

---

## Visualizar Experimentos no MLflow UI

```bash
mlflow ui
```

Acesse `http://localhost:5000` no navegador para comparar:
- Parâmetros de cada experimento
- Métricas (F1-Score, ROC-AUC, tempo de treino)
- Artefatos dos modelos treinados

---

## Executar a Interface Streamlit

```bash
streamlit run app/app.py
```

O app carrega o modelo campeão automaticamente via `models/champion_run_id.txt`.

Para forçar um modelo específico (ex: o com PCA):

```bash
MODEL_URI=runs:/<run_id>/model streamlit run app/app.py
```

---

## Experimentos MLflow — Descrição Técnica

### Experimento 1: RF Baseline (sem redução)
Random Forest com `class_weight="balanced"`, GridSearchCV 5-fold, otimizando F1-Score.
Serve como referência: com apenas 10 features, não há maldição da dimensionalidade.

### Experimento 2: RF + PCA
`PCA(n_components=0.95)` — sklearn seleciona automaticamente o número mínimo de componentes
para explicar 95% da variância. Com correlações entre as features de atraso (30-59, 60-89, 90+
dias), esperam-se ~7-8 componentes.

**Trade-off:** reduz ruído potencial, mas perde interpretabilidade das features originais
(relevante para LGPD Art. 20 e resoluções BACEN).

### Experimento 3: RF + LDA
`LDA(n_components=1)` — limitação matemática de classificação binária: máximo de
`min(n_classes - 1, n_features) = 1` componente discriminante. Todo o espaço de 10
dimensões é projetado em 1 único eixo. Espera-se queda no F1 vs. baseline.

**Vantagem:** LDA é supervisionado — o componente maximiza a separabilidade entre classes.

### Experimento 4: Decision Tree Baseline
Comparação com árore de decisão regularizada (GridSearchCV). Permite discutir o
trade-off entre interpretabilidade total (regras auditáveis) e performance do ensemble.

---

## Decisões de Qualidade de Dados

| Problema | Tratamento | Justificativa |
|----------|------------|---------------|
| `MonthlyIncome` ausente (~20%) | `SimpleImputer(strategy='median')` | Mediana robusta a outliers |
| `NumberOfDependents` ausente (~2.5%) | `SimpleImputer(strategy='median')` | Distribuição assimétrica |
| `DebtRatio` outliers (valores > 1000) | Capping no percentil 99 | Erros de dados / casos extremos |
| `RevolvingUtilizationOfUnsecuredLines` outliers | Capping no percentil 99 | Valores fisicamente impossíveis |
| Classe positiva ~6.7% (desbalanceada) | `class_weight="balanced"` | Preserva recall da classe minoritária |

---

## Métricas Técnicas e de Negócio

| Métrica | Tipo | Relevância |
|---------|------|------------|
| F1-Score | Técnica | Balanceia precision e recall — ideal para classes desbalanceadas |
| ROC-AUC | Técnica | Capacidade discriminativa independente do threshold |
| Falso Negativo (FN) | Negócio | Aprovação de cliente que vai inadimplir → prejuízo financeiro |
| Falso Positivo (FP) | Negócio | Reprovação de bom pagador → perda de receita, risco reputacional |
