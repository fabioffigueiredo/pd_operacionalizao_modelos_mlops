"""
app.py
------
Interface Streamlit para inferência do modelo campeão de scoring de crédito.

PD2 MLOps — Instituto Infnet
Aluno: Fabio Ferreira Figueiredo

Uso:
    cd /caminho/para/PD\ 2\ MLops
    streamlit run app/app.py

Pré-requisito: executar `python src/train.py` para gerar o modelo e o arquivo
models/champion_run_id.txt antes de iniciar o Streamlit.
"""

import os
import sys
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Scoring de Crédito | PD2 MLOps",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constantes ──────────────────────────────────────────────────────────────

FEATURES = [
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

FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Utilização de Crédito Rotativo",
    "age": "Idade (anos)",
    "NumberOfTime30-59DaysPastDueNotWorse": "Atrasos 30–59 dias",
    "DebtRatio": "Razão de Endividamento",
    "MonthlyIncome": "Renda Mensal (USD)",
    "NumberOfOpenCreditLinesAndLoans": "Linhas de Crédito Abertas",
    "NumberOfTimes90DaysLate": "Atrasos acima de 90 dias",
    "NumberRealEstateLoansOrLines": "Empréstimos Imobiliários",
    "NumberOfTime60-89DaysPastDueNotWorse": "Atrasos 60–89 dias",
    "NumberOfDependents": "Número de Dependentes",
}

FEATURE_HELP = {
    "RevolvingUtilizationOfUnsecuredLines":
        "Percentual do limite de crédito rotativo utilizado (0 = sem uso, 1 = 100% utilizado). "
        "Valores acima de 0.9 são sinal de alto risco.",
    "age": "Idade do solicitante em anos completos. Mínimo: 18.",
    "NumberOfTime30-59DaysPastDueNotWorse":
        "Quantas vezes o cliente ficou entre 30 e 59 dias em atraso nos últimos 2 anos.",
    "DebtRatio":
        "Pagamentos mensais de dívidas dividido pela renda mensal bruta. "
        "Valores > 1.0 significam que as dívidas superam a renda.",
    "MonthlyIncome":
        "Renda bruta mensal em dólares americanos. Deixe 0 se desconhecida "
        "(será imputada pela mediana do modelo).",
    "NumberOfOpenCreditLinesAndLoans":
        "Número total de linhas de crédito e empréstimos atualmente abertos.",
    "NumberOfTimes90DaysLate":
        "Número de vezes que o cliente ficou com atraso acima de 90 dias. "
        "Forte preditor de inadimplência futura.",
    "NumberRealEstateLoansOrLines":
        "Número de empréstimos ou linhas de crédito imobiliário (hipotecas, financiamentos).",
    "NumberOfTime60-89DaysPastDueNotWorse":
        "Quantas vezes o cliente ficou entre 60 e 89 dias em atraso nos últimos 2 anos.",
    "NumberOfDependents":
        "Número de dependentes na família, excluindo cônjuge.",
}

FEATURE_DEFAULTS = {
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
}


# ─── Carregamento do modelo ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo campeão...")
def load_champion_model():
    """
    Carrega o modelo campeão do MLflow local com fallback progressivo.

    Estratégia de carregamento:
      1. Variável de ambiente MODEL_URI (ex: MODEL_URI=runs:/<run_id>/model)
      2. Arquivo models/champion_run_id.txt gerado automaticamente por train.py
      3. Erro claro com instruções de como resolver

    @st.cache_resource garante que o modelo é carregado uma única vez por
    sessão do servidor, mesmo com múltiplos usuários simultâneos.
    """
    # Resolver tracking URI relativo ao diretório de execução
    project_root = Path(__file__).parent.parent
    mlruns_path = project_root / "mlruns"
    mlflow.set_tracking_uri(str(mlruns_path))

    # Opção 1: variável de ambiente explícita
    model_uri = os.environ.get("MODEL_URI")

    # Opção 2: arquivo gerado por train.py
    if not model_uri:
        run_id_file = project_root / "models" / "champion_run_id.txt"
        if run_id_file.exists():
            run_id = run_id_file.read_text().strip()
            model_uri = f"runs:/{run_id}/model"
        else:
            st.error(
                "**Modelo não encontrado.** Execute primeiro:\n\n"
                "```bash\ncd 'PD 2 MLops'\npython src/train.py\n```\n\n"
                "Ou defina a variável de ambiente:\n\n"
                "```bash\nMODEL_URI=runs:/<run_id>/model streamlit run app/app.py\n```"
            )
            st.stop()

    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model, model_uri
    except Exception as e:
        st.error(
            f"**Erro ao carregar modelo** `{model_uri}`:\n\n```\n{e}\n```\n\n"
            "Verifique se o treinamento foi concluído e o arquivo `mlruns/` está acessível."
        )
        st.stop()


# ─── Helpers de UI ───────────────────────────────────────────────────────────

def render_resultado(pred: int, proba: np.ndarray):
    """Renderiza o card de resultado com cores e probabilidades."""
    prob_inadimplente = proba[1]
    prob_adimplente = proba[0]

    if pred == 0:
        st.markdown(
            """
            <div style="background:#e8f5e9;border-left:6px solid #2e7d32;
                        padding:20px;border-radius:8px;margin-bottom:16px">
                <h2 style="color:#1b5e20;margin:0">✅ APROVADO — Baixo Risco</h2>
                <p style="color:#2e7d32;margin:8px 0 0">
                    O modelo classifica este solicitante como <b>adimplente</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:#ffebee;border-left:6px solid #c62828;
                        padding:20px;border-radius:8px;margin-bottom:16px">
                <h2 style="color:#7f0000;margin:0">❌ REPROVADO — Alto Risco</h2>
                <p style="color:#c62828;margin:8px 0 0">
                    O modelo classifica este solicitante como <b>potencialmente inadimplente</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Métricas de probabilidade
    col_a, col_b = st.columns(2)
    col_a.metric("Probabilidade: Adimplente", f"{prob_adimplente:.1%}")
    col_b.metric("Probabilidade: Inadimplente", f"{prob_inadimplente:.1%}")

    # Gráfico de barras
    fig = go.Figure(go.Bar(
        x=["Adimplente (Classe 0)", "Inadimplente (Classe 1)"],
        y=[prob_adimplente, prob_inadimplente],
        marker_color=["#43a047", "#e53935"],
        text=[f"{prob_adimplente:.1%}", f"{prob_inadimplente:.1%}"],
        textposition="outside",
        textfont=dict(size=14),
    ))
    fig.update_layout(
        title="Distribuição de Probabilidade da Predição",
        yaxis=dict(range=[0, 1.15], tickformat=".0%", title="Probabilidade"),
        xaxis=dict(title="Classe"),
        height=360,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(model, has_reduction: bool, reduction_name: str):
    """Renderiza gráfico de importância de features (quando disponível)."""
    clf_step = model.named_steps.get("clf")
    has_importances = clf_step is not None and hasattr(clf_step, "feature_importances_")

    if has_reduction:
        st.info(
            f"**Interpretabilidade limitada — modelo com {reduction_name}**\n\n"
            f"As features originais foram transformadas em componentes latentes pelo {reduction_name}. "
            f"As importâncias do Random Forest refletem os componentes, não as features originais. "
            f"Consulte o relatório técnico para análise detalhada do impacto da redução de dimensionalidade."
        )
    elif has_importances:
        importances = clf_step.feature_importances_
        imp_df = (
            pd.DataFrame({"feature": FEATURES, "importance": importances})
            .sort_values("importance", ascending=True)
        )
        labels = [FEATURE_LABELS.get(f, f) for f in imp_df["feature"]]

        fig2 = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=labels,
            orientation="h",
            marker_color="#1565c0",
            text=[f"{v:.3f}" for v in imp_df["importance"]],
            textposition="outside",
        ))
        fig2.update_layout(
            title="Importância das Features (Random Forest — MDI)",
            height=400,
            xaxis=dict(title="Importância (Mean Decrease Impurity)"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Importâncias de features não disponíveis para este tipo de modelo.")


# ─── Layout principal ─────────────────────────────────────────────────────────

st.title("🏦 Sistema de Scoring de Crédito")
st.caption(
    "**PD2 MLOps — Instituto Infnet** | "
    "Modelo: Random Forest com class_weight='balanced' | "
    "Dataset: Give Me Some Credit (150.000 registros)"
)
st.divider()

# Carregar modelo
model, model_uri = load_champion_model()

# Detectar tipo de redução no pipeline
has_pca = "pca" in model.named_steps
has_lda = "lda" in model.named_steps
has_reduction = has_pca or has_lda
reduction_name = "PCA" if has_pca else ("LDA" if has_lda else "Nenhuma")

# Info do modelo no topo
with st.expander("ℹ️ Informações do Modelo Carregado", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("URI do Modelo", model_uri.split("/model")[0].split("runs:/")[-1][:16] + "...")
    col2.metric("Redução Dimensional", reduction_name)
    clf_step = model.named_steps.get("clf")
    if clf_step:
        col3.metric("Algoritmo", type(clf_step).__name__)
    st.code(f"MODEL_URI = {model_uri}", language="bash")

# ─── Sidebar — Inputs do Solicitante ─────────────────────────────────────────
st.sidebar.header("📋 Dados do Solicitante")
st.sidebar.markdown("Preencha os campos e clique em **Avaliar**.")
st.sidebar.divider()

inputs = {}

inputs["RevolvingUtilizationOfUnsecuredLines"] = st.sidebar.slider(
    FEATURE_LABELS["RevolvingUtilizationOfUnsecuredLines"],
    min_value=0.0, max_value=1.0,
    value=float(FEATURE_DEFAULTS["RevolvingUtilizationOfUnsecuredLines"]),
    step=0.01,
    help=FEATURE_HELP["RevolvingUtilizationOfUnsecuredLines"],
)
inputs["age"] = st.sidebar.number_input(
    FEATURE_LABELS["age"],
    min_value=18, max_value=100,
    value=int(FEATURE_DEFAULTS["age"]),
    step=1,
    help=FEATURE_HELP["age"],
)
inputs["NumberOfTime30-59DaysPastDueNotWorse"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfTime30-59DaysPastDueNotWorse"],
    min_value=0, max_value=20,
    value=int(FEATURE_DEFAULTS["NumberOfTime30-59DaysPastDueNotWorse"]),
    step=1,
    help=FEATURE_HELP["NumberOfTime30-59DaysPastDueNotWorse"],
)
inputs["DebtRatio"] = st.sidebar.number_input(
    FEATURE_LABELS["DebtRatio"],
    min_value=0.0, max_value=10.0,
    value=float(FEATURE_DEFAULTS["DebtRatio"]),
    step=0.01,
    format="%.2f",
    help=FEATURE_HELP["DebtRatio"],
)
inputs["MonthlyIncome"] = st.sidebar.number_input(
    FEATURE_LABELS["MonthlyIncome"],
    min_value=0.0, max_value=500_000.0,
    value=float(FEATURE_DEFAULTS["MonthlyIncome"]),
    step=100.0,
    format="%.0f",
    help=FEATURE_HELP["MonthlyIncome"],
)
inputs["NumberOfOpenCreditLinesAndLoans"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfOpenCreditLinesAndLoans"],
    min_value=0, max_value=50,
    value=int(FEATURE_DEFAULTS["NumberOfOpenCreditLinesAndLoans"]),
    step=1,
    help=FEATURE_HELP["NumberOfOpenCreditLinesAndLoans"],
)
inputs["NumberOfTimes90DaysLate"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfTimes90DaysLate"],
    min_value=0, max_value=20,
    value=int(FEATURE_DEFAULTS["NumberOfTimes90DaysLate"]),
    step=1,
    help=FEATURE_HELP["NumberOfTimes90DaysLate"],
)
inputs["NumberRealEstateLoansOrLines"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberRealEstateLoansOrLines"],
    min_value=0, max_value=20,
    value=int(FEATURE_DEFAULTS["NumberRealEstateLoansOrLines"]),
    step=1,
    help=FEATURE_HELP["NumberRealEstateLoansOrLines"],
)
inputs["NumberOfTime60-89DaysPastDueNotWorse"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfTime60-89DaysPastDueNotWorse"],
    min_value=0, max_value=20,
    value=int(FEATURE_DEFAULTS["NumberOfTime60-89DaysPastDueNotWorse"]),
    step=1,
    help=FEATURE_HELP["NumberOfTime60-89DaysPastDueNotWorse"],
)
inputs["NumberOfDependents"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfDependents"],
    min_value=0, max_value=20,
    value=int(FEATURE_DEFAULTS["NumberOfDependents"]),
    step=1,
    help=FEATURE_HELP["NumberOfDependents"],
)

st.sidebar.divider()
avaliar = st.sidebar.button(
    "🔍 Avaliar Risco de Crédito",
    type="primary",
    use_container_width=True,
)

# ─── Área Principal ───────────────────────────────────────────────────────────

if avaliar:
    df_input = pd.DataFrame([inputs])[FEATURES]

    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]

    col_result, col_import = st.columns([1, 1])

    with col_result:
        st.subheader("Resultado da Análise")
        render_resultado(int(pred), proba)

    with col_import:
        st.subheader("Explicabilidade do Modelo")
        render_feature_importance(model, has_reduction, reduction_name)

    # Tabela de dados inseridos
    with st.expander("📊 Dados inseridos para esta análise", expanded=False):
        df_display = pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "Valor": [inputs[f] for f in FEATURES],
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)

else:
    # Estado inicial — sem análise ainda
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Como usar")
        st.markdown("""
        1. **Preencha os dados** do solicitante na barra lateral esquerda
        2. **Clique em Avaliar** para obter a predição
        3. O resultado aparece aqui com **probabilidades** e **explicabilidade**

        ### Sobre o Modelo
        | Atributo | Valor |
        |----------|-------|
        | Algoritmo | Random Forest |
        | Balanceamento | class_weight="balanced" |
        | Métrica principal | F1-Score |
        | Dataset de treino | 120.000 registros |
        | Rastreamento | MLflow |

        ### Contexto Regulatório
        Este sistema foi desenvolvido considerando:
        - **LGPD Art. 20**: direito à explicação de decisões automatizadas
        - **Resoluções BACEN**: requisitos de transparência em modelos de crédito
        """)

    with col_r:
        st.subheader("Perfil de Risco — Referência")
        st.markdown("""
        | Indicador | Baixo Risco | Alto Risco |
        |-----------|-------------|------------|
        | Utilização crédito rotativo | < 30% | > 80% |
        | Atrasos > 90 dias | 0 | ≥ 1 |
        | Razão de endividamento | < 0.4 | > 0.8 |
        | Atrasos 30–59 dias | 0 | ≥ 2 |

        > **Nota:** Estes são indicadores de referência baseados na análise
        > exploratória do dataset. O modelo considera a interação entre
        > todas as features simultaneamente.
        """)

        st.info(
            "**Aviso:** Este sistema é para fins acadêmicos (PD2 MLOps — Instituto Infnet). "
            "Não deve ser utilizado para decisões de crédito reais sem validação regulatória."
        )
