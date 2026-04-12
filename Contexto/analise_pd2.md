
# Análise Projetod de disciplina 2

Olá, Fabio! Excelente trabalho no seu primeiro Projeto de Disciplina. Analisando o seu repositório e o PDF entregue, você construiu uma base muito sólida com o dataset _Give Me Some Credit_. O uso do Perceptron como baseline, a otimização da Árvore de Decisão e a vitória do Random Forest (F1-Score de 0.4328) com tratamento de desbalanceamento (`class_weight="balanced"`) foram excelentes decisões técnicas.

Agora, na **Parte 2 (MLOps)**, a mentalidade muda. O professor Ícaro deixou claro que o foco deixa de ser o "modelo isolado" e passa a ser o **sistema de machine learning**. Você deixará de ser apenas um Cientista de Dados focado em métricas para atuar como um **Engenheiro de Machine Learning**, focando em reprodutibilidade, rastreamento, controle de complexidade e deploy.

O professor frisou que **o código funcional isoladamente não é suficiente; a avaliação considera estrutura, rastreabilidade e decisões técnicas**.

Abaixo, estruturei o guia passo a passo para a sua segunda entrega, aproveitando o código do seu primeiro projeto, além dos trechos de código fundamentais que você precisará implementar.

----------

### Passo 1: Reestruturação do Projeto (Saindo do Notebook)

Na primeira fase, você usou o `projeto_credito_supervisionado.ipynb`. Agora, o professor exige que a lógica principal saia do Jupyter Notebook exploratório e vá para **scripts Python modulares (`.py`)**. O Jupyter deve servir apenas para visualização e EDA.

**Nova Estrutura Sugerida para o seu GitHub:**

```
pd-ml-scikit-learning/
│
├── config/
│   └── pipeline.yaml          # Arquivo para não 'chumbar' hiperparâmetros no código
├── src/
│   ├── data_ingestion.py      # Script para baixar do Kaggle e tratar nulos
│   ├── preprocessing.py       # Transformadores customizados (SimpleImputer, Scaler)
│   ├── train_mlflow.py        # Treinamento com MLflow e Redução de Dimensionalidade
│   └── evaluate.py            # Avaliação de métricas no holdout (teste)
├── app/
│   └── app_streamlit.py       # Simulação da interface do modelo em produção
└── relatorio_final_mlops.pdf  # O relatório exigido pela rubrica

```

----------

### Passo 2: Rastreamento de Experimentos com MLflow

No primeiro projeto, você usou o `GridSearchCV` e printou os resultados. Agora, você deve registrar todos os experimentos (parâmetros, métricas F1/ROC-AUC e o próprio modelo) no **MLflow**.

**Adaptação do seu código para o MLflow (`src/train_mlflow.py`):**

```
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
# Importe seu preprocessor e os dados divididos (X_train, y_train, etc.)

def treinar_random_forest_mlflow(X_train, X_test, y_train, y_test, max_depth, n_estimators):

    # Inicia o rastreamento do MLflow
    with mlflow.start_run(run_name="RandomForest_Baseline"):

        # 1. Registra os parâmetros que você está testando
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("class_weight", "balanced")

        # 2. Configura e treina o modelo que ganhou no seu PD1
        clf = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=42
        )

        # Integra com o seu preprocessor existente (Imputer + Scaler)
        pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", clf)
        ])

        pipe.fit(X_train, y_train)

        # 3. Calcula as métricas
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        # 4. Registra as métricas e o modelo no MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)
        mlflow.sklearn.log_model(pipe, "modelo_rf_credito")

        print(f"Modelo salvo no MLflow com F1: {f1:.4f} e ROC: {roc:.4f}")

# Exemplo de execução
treinar_random_forest_mlflow(X_train, X_test, y_train, y_test, max_depth=8, n_estimators=100)

```

----------

### Passo 3: Redução de Dimensionalidade (Obrigatório no PD 2)

Esta é uma exigência estrita da Fase 4 do novo documento: você precisa escolher e aplicar **duas técnicas de redução de dimensionalidade** (ex: PCA e t-SNE, ou PCA e LDA) ao seu pipeline de modelagem.

Você deve comparar o seu Random Forest _sem redução_ (o que você fez no PD1) com o Random Forest _com redução_, analisando o impacto no F1-Score, no custo de processamento e na explicabilidade (que você citou como vital para LGPD/BACEN).

**Adaptação do seu código adicionando PCA:**

```
from sklearn.decomposition import PCA

# Você vai adicionar o PCA logo após o preprocessor no seu Pipeline
pipeline_pca = Pipeline([
    ("prep", preprocessor),          # Seu Imputer + Scaler original
    ("pca", PCA(n_components=5)),    # Reduzindo as 10 features financeiras para 5 componentes principais
    ("clf", RandomForestClassifier(class_weight="balanced", max_depth=8, random_state=42))
])

# Para o MLflow, você registraria:
with mlflow.start_run(run_name="RandomForest_com_PCA"):
    mlflow.log_param("reducao_dimensionalidade", "PCA")
    mlflow.log_param("n_components", 5)

    pipeline_pca.fit(X_train, y_train)
    # ... calcular e logar métricas como no código acima

```

_Atenção para o Relatório:_ No PD1, você notou que o _RevolvingUtilization_ era o nó raiz e o preditor mais forte. Ao aplicar PCA, você perderá essa interpretabilidade direta, pois as features virarão componentes genéricos. Isso é um excelente ponto de discussão crítica exigido pelo professor para o relatório final.

----------

### Passo 4: Operacionalização e Simulação de Produção (Deploy)

A parte final exige empacotar o modelo como um artefato e expô-lo em um serviço de inferência, simulando produção. O professor citou explicitamente o uso do **Streamlit** nas aulas para fazer essa demonstração de forma rápida.

Você precisará criar um script curto (`app_streamlit.py`) que carregue o modelo do MLflow (ou via arquivo `.pkl`) e permita que o usuário (ou você, no vídeo de demonstração) insira os dados de um cliente (idade, salário, dependentes, etc.) para ver se o crédito será aprovado ou se ele tem risco de calote.

**Código base para o seu Streamlit (`app_streamlit.py`):**

```
import streamlit as st
import pandas as pd
import mlflow.sklearn

st.title("Sistema de Scoring de Crédito - BACEN/LGPD")

# 1. Carregar o modelo salvo pelo MLflow
@st.cache_resource
def load_model():
    # Substitua pela URI gerada pelo seu MLflow (ex: runs:/<run_id>/modelo_rf_credito)
    return mlflow.sklearn.load_model("models:/CreditoModel/Production")

modelo = load_model()

# 2. Formulário para entrada de dados do cliente
st.sidebar.header("Dados do Cliente")
age = st.sidebar.number_input("Idade", min_value=18, max_value=100, value=40)
monthly_income = st.sidebar.number_input("Renda Mensal (USD)", value=5000)
debt_ratio = st.sidebar.number_input("Taxa de Endividamento (DebtRatio)", value=0.3)
revolving_util = st.sidebar.number_input("Utilização de Linhas Rotativas", value=0.5)
# ... adicionar os inputs para as outras features necessárias do seu X_train

if st.sidebar.button("Avaliar Risco de Crédito"):
    # 3. Montar o DataFrame de inferência
    dados_cliente = pd.DataFrame({
        "age": [age],
        "MonthlyIncome": [monthly_income],
        "DebtRatio": [debt_ratio],
        "RevolvingUtilizationOfUnsecuredLines": [revolving_util]
        # ... adicionar as outras features preenchidas com 0 ou mediana para teste
    })

    # 4. Fazer a predição
    predicao = modelo.predict(dados_cliente)
    proba = modelo.predict_proba(dados_cliente)

    st.subheader("Resultado da Análise")
    if predicao == 1:
        st.error(f"ALTO RISCO DE INADIMPLÊNCIA (Probabilidade: {proba:.2%})")
    else:
        st.success(f"CRÉDITO APROVADO - Baixo Risco (Probabilidade de calote: {proba:.2%})")

```

----------

### O que entregar ao Professor (Checklist Final):

1.  **Repositório GitHub organizado** com os arquivos em pastas `.py` e não apenas notebooks.
2.  **O Relatório Final (PDF)**: Deve conter os links do GitHub, explicar a reestruturação dos dados e qualidade (missing values e os outliers absurdos de `DebtRatio` que você achou no PD1), descrever a experiência da Redução de Dimensionalidade (PCA vs sem PCA) e justificar qual modelo foi eleito como "Campeão" para ir para o Streamlit.
3.  **Vídeo de Demonstração Funcional:** Um vídeo curto gravando a tela do seu computador mostrando o seu painel do MLflow com os rastreamentos e depois abrindo a tela do Streamlit e clicando no botão para avaliar o risco de um cliente.

Sua fundação metodológica já está aprovada, agora o trabalho é apenas transformá-la de "pesquisa" para um "software escalável". Se precisar de ajuda para escrever os arquivos YAML de configuração ou aprofundar os logs do Great Expectations (ferramenta de qualidade de dados citada pelo professor), me avise!
