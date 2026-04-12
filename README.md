<div align="center">
  <h1>
    <img src="pd-ml-scikit-learning-main/images/logo_infnet.png" alt="Instituto Infnet" width="80" title="Instituto Infnet" align="absmiddle"/>
    Projeto de Disciplina: Operacionalização de Modelos com MLOps
  </h1>
</div>

<div align="center">

  **Pós-Graduação em Machine Learning, Deep Learning e Inteligência Artificial**<br>
  **Disciplina:** Operacionalização de Modelos com MLOps<br>
  **Professor:** Ícaro Augusto Maccari Zelioli<br>
  **Aluno:** Fabio Ferreira Figueiredo <a href="https://github.com/fabioffigueiredo"><img src="https://img.shields.io/badge/GitHub-perfil-black?logo=github" alt="GitHub"></a>

  <p>
    <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2?style=flat-square" alt="MLflow">
    <img src="https://img.shields.io/badge/Streamlit-infer%C3%AAncia-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  </p>
</div>

---

## Visão Geral do Projeto

Este repositório representa o **PD2** da disciplina, transformando o trabalho exploratório do PD1 em um fluxo mais próximo de engenharia de machine learning: código modular, rastreamento de experimentos com `MLflow`, seleção de modelo campeão e inferência via `Streamlit`.

O problema continua sendo o mesmo: prever **inadimplência severa** (`SeriousDlqin2yrs`) no dataset **Give Me Some Credit**, com aproximadamente 150 mil registros e classe positiva em torno de 6,7%.

O foco desta etapa não está apenas em métrica, mas em demonstrar:
- estrutura de projeto adequada à prática de engenharia
- pipeline de dados e features com decisões explícitas
- experimentos reprodutíveis e rastreáveis
- operacionalização com inferência e visão de monitoramento

---

## Relação com o PD1

Este projeto parte do trabalho anterior em:

- [PD1 — Fundamentos de ML com Scikit-Learn](https://github.com/fabioffigueiredo/pd-ml-scikit-learning)
- Material legado local em [pd-ml-scikit-learning-main](pd-ml-scikit-learning-main)

No PD1, o foco foi comparar modelos supervisionados em notebook. No PD2, a entrega evolui para um sistema modular com `train.py`, configuração centralizada, rastreabilidade no `MLflow` e interface de inferência em `Streamlit`.

---

## Estrutura do Repositório

```text
PD 2 MLops/
├── app/
│   └── app.py
├── config/
│   └── pipeline.yaml
├── Contexto/
│   ├── Contexto PD 2 Operacionalização de Modelos com MLOps.md
│   ├── analise_pd2.md
│   └── artefatos_locais/        # apoio local e rascunhos, fora do GitHub
├── data/
│   └── raw/
├── models/
├── pd-ml-scikit-learning-main/
├── reports/
│   ├── relatorio_tecnico.md
│   └── relatorio_tecnico.pdf
├── scripts/
│   └── render_relatorio_pdf.py
├── src/
│   ├── data_processing.py
│   └── train.py
├── requirements.txt
└── README.md
```

---

## Resultados Principais

| Experimento | Redução | F1-Score | ROC-AUC | Observação |
|---|---|---:|---:|---|
| `RF_com_PCA` | PCA (9 componentes) | **0.4354** | 0.8555 | Campeão factual carregado no app |
| `RF_sem_reducao_baseline` | Nenhuma | 0.4308 | **0.8572** | Melhor interpretabilidade |
| `RF_com_LDA` | LDA (1 componente) | 0.3494 | 0.8156 | Perda forte de informação |
| `DT_sem_reducao_baseline` | Nenhuma | 0.3408 | 0.8544 | Regras auditáveis, menor F1 |

**Leitura de engenharia:** o `RF + PCA` venceu pela métrica primária, mas o `RF baseline` continua tecnicamente defensável quando a prioridade é interpretabilidade em contexto regulado.

---

## Como Executar o Projeto Localmente

### 1. Pré-requisitos

- Python `3.11+`
- Dataset `cs-training.csv`

### 2. Criar ambiente e instalar dependências

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Disponibilizar o dataset

```bash
mkdir -p data/raw
cp /caminho/para/cs-training.csv data/raw/
```

Se quiser reutilizar a base do PD1:

```bash
cp pd-ml-scikit-learning-main/archive/cs-training.csv data/raw/
```

### 4. Rodar o treinamento

```bash
python src/train.py
```

Ao final, o projeto salva automaticamente o `run_id` do campeão em `models/champion_run_id.txt`.

### 5. Abrir o MLflow

```bash
mlflow ui
```

Depois, acesse `http://127.0.0.1:5000`.

### 6. Abrir a interface de inferência

```bash
streamlit run app/app.py
```

O `Streamlit` lê o `run_id` campeão salvo e carrega o modelo correspondente do `MLflow`.

---

## O Que Foi Entregue

- Repositório modularizado, reduzindo dependência de notebook
- Quatro experimentos comparativos rastreados no `MLflow`
- Modelo campeão persistido e reaproveitado na inferência
- Interface `Streamlit` para simulação de operação
- Relatório técnico em Markdown e PDF:
  - [Relatório Técnico em PDF](reports/relatorio_tecnico.pdf)
  - [Relatório Técnico em Markdown](reports/relatorio_tecnico.md)

---

## Observações Sobre Versionamento

- Artefatos locais de treino, logs, vídeos e rascunhos foram isolados para não poluir o repositório.
- O que não precisa ir para o GitHub fica em `Contexto/artefatos_locais/` e está ignorado no `.gitignore`.

---

<div align="center">
  <small>Desenvolvido para fins acadêmicos e de demonstração técnica.<br>Abril / 2026</small>
</div>
