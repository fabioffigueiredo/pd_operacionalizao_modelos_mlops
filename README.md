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

Repositório desta entrega: [pd_operacionalizao_modelos_mlops](https://github.com/fabioffigueiredo/pd_operacionalizao_modelos_mlops)

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

Se você quiser reproduzir apenas o material legado em notebook do PD1, use as dependências próprias em [pd-ml-scikit-learning-main/requirements.txt](pd-ml-scikit-learning-main/requirements.txt). O `requirements.txt` da raiz cobre o fluxo ativo do PD2.

---

## Estrutura do Repositório

```text
PD 2 MLops/
├── app/
│   └── app.py                      # Interface Streamlit de inferência
├── config/
│   └── pipeline.yaml               # Configuração centralizada (fonte única de verdade)
├── data/
│   └── raw/                        # Dataset cs-training.csv (não versionado)
├── mlflow+streamlit_mlops.mp4      # Vídeo de demonstração
├── mlruns/                         # Artefatos MLflow (gerado automaticamente)
├── models/
│   └── champion_run_id.txt         # run_id do modelo campeão
├── pd-ml-scikit-learning-main/     # Material legado do PD1
├── reports/
│   ├── relatorio_tecnico.md        # Relatório técnico (fonte)
│   ├── relatorio_tecnico.pdf       # Relatório técnico (renderizado)
│   ├── metrics_extended.csv        # Precision, Recall e confusion matrix por modelo
│   └── pca_loadings.csv            # Loadings das componentes PCA
├── scripts/
│   ├── extract_metrics.py          # Extrai métricas dos modelos salvos no MLflow
│   └── render_relatorio_pdf.py     # Gera o PDF a partir do Markdown
├── src/
│   ├── data_processing.py          # Ingestão, outlier capping e preprocessor factory
│   └── train.py                    # Orquestração dos 4 experimentos MLflow
├── requirements.txt
└── README.md
```

---

## Resultados Principais

| Experimento | Redução | Precision | Recall | F1-Score | ROC-AUC | Observação |
|---|---|---:|---:|---:|---:|---|
| `RF_com_PCA` ⭐ | PCA (9 componentes) | 0.3595 | **0.5382** | **0.4354** | 0.8555 | Campeão — maior F1 e Recall |
| `RF_sem_reducao_baseline` | Nenhuma | **0.3893** | 0.4823 | 0.4308 | **0.8572** | Maior Precision, melhor interpretabilidade |
| `RF_com_LDA` | LDA (1 componente) | 0.2460 | 0.6010 | 0.3494 | 0.8156 | Perda forte de informação |
| `DT_sem_reducao_baseline` | Nenhuma | 0.2211 | 0.7426 | 0.3408 | 0.8544 | Maior Recall, muitos falsos alarmes |

**Leitura de engenharia:** o `RF + PCA` venceu pela métrica primária (F1) e detecta mais inadimplentes (Recall 53,8%). O `RF baseline` tem a maior Precision — menos recusas indevidas — e continua tecnicamente defensável em contexto regulado onde interpretabilidade é prioritária.

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
- Vídeo de demonstração:
  - [Arquivo no repositório](mlflow+streamlit_mlops.mp4)
  - [Backup no Google Drive](https://drive.google.com/file/d/11Yn6D01kEwuc6N-__t40ZlxxyOoEa-uM/view?usp=sharing)
- Relatório técnico em Markdown e PDF:
  - [Relatório Técnico em PDF](reports/relatorio_tecnico.pdf)
  - [Relatório Técnico em Markdown](reports/relatorio_tecnico.md)
  - Repositório no GitHub: [pd_operacionalizao_modelos_mlops](https://github.com/fabioffigueiredo/pd_operacionalizao_modelos_mlops)

---

## Observações Sobre Versionamento

- Artefatos temporários de treino, logs e arquivos de renderização local ficam fora do Git.
- O HTML intermediário do PDF é gerado em `.render_tmp/`, que está ignorado no `.gitignore`.

---

<div align="center">
  <small>Desenvolvido para fins acadêmicos e de demonstração técnica.<br>Abril / 2026</small>
</div>
