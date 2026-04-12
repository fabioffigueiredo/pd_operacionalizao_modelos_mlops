<div align="center">
  <h1>
    <img src="images/logo_infnet.png" alt="Instituto Infnet" width="80" title="Instituto Infnet" align="absmiddle"/> 
    Projeto de Disciplina: Avaliação de Risco de Crédito
  </h1>
</div>

<div align="center">

  **Pós-Graduação em Machine Learning, Deep Learning e Inteligência Artificial**<br>
  **Disciplina:** Fundamentos de Machine Learning com Scikit-Learn<br>
  **Professor:** Icaro Augusto Maccari Zelioli<br>
  **Alunos:** Fabio Ferreira Figueiredo <a href="https://github.com/fabioffigueiredo"><img src="https://img.shields.io/badge/GitHub-repo-black?logo=github" alt="GitHub"></a> 
  Felipe Moreira Szczpanski <a href="https://github.com/szczpanski"><img src="https://img.shields.io/badge/GitHub-repo-black?logo=github" alt="GitHub"></a>
  Lauro Camilo Barbosa Marques da Rocha <a href="https://github.com/LMRocha"><img src="https://img.shields.io/badge/GitHub-repo-black?logo=github" alt="GitHub"></a>

  <p>
    <img src="https://img.shields.io/badge/python-v._3.13-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/jupyter-v._5.9-blue?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
    <img src="https://img.shields.io/badge/scikit--learn-1.8.0-orange?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-Learn">
  </p>
</div>

---

## Visão Geral do Projeto

Este projeto avalia o risco de inadimplência de tomadores de crédito utilizando o dataset **Give Me Some Credit (Kaggle)**. A modelagem preditiva foi desenvolvida do zero empregando algoritmos como *Perceptron*, *Árvores de Decisão* e *Random Forest* da biblioteca `scikit-learn`.

O projeto aplica aprendizado de máquina para capturar correlações não lineares. A abordagem segue uma progressão lógica de complexidade: desde um baseline linear interpretável até ensembles de alta performance, sempre priorizando a explicabilidade exigida pelo contexto regulatório financeiro (LGPD e BACEN).

O objetivo principal é prever se um cliente incorrerá em uma inadimplência grave (atraso nas obrigações financeiras por 90 dias ou mais) nos próximos dois anos, utilizando variáveis demográficas, financeiras e comportamentais. 

---

## O Dataset: Give Me Some Credit

A base de dados utilizada provém de uma [competição do Kaggle (brycecf/give-me-some-credit-dataset)](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) elaborada para construir algoritmos que definem quem pode ou não sofrer dificuldades financeiras no curto/médio prazo. O problema é formulado como uma classificação binária sobre o dataset de aproximadamente 150.000 registros.

### Desafios Técnicos do Dataset
Os dados financeiros reais trazem características que tornam a modelagem mais complexa:
- **Desbalanceamento Extremo**: Apenas cerca de 6,7% dos registros representam clientes inadimplentes (classe positiva), foi utilizado F1-Score e class_weight="balanced" para mitigar o viés da maioria.
- **Valores Ausentes**: Volumes de dados faltantes acentuados, especialmente nas features `MonthlyIncome` (~20%) e `NumberOfDependents` (~2,5%).
- **Outliers**: Presença de exceções em dados financeiros (ex: utilização de cartão acima de 100%, ou taxas de endividamento fora do comum, tratados via pipelines de pré-processamento estatístico.

O projeto aplica técnicas robustas de engenharia de features, preenchimento estatístico via Pipeline (`SimpleImputer` com medianas), e hiperparametrização via `GridSearchCV` focada na métrica estrutural de **F1-Score**, balanceando *Recall* e *Precision* frente à minoria de casos positivos.

---

## O que o Notebook Faz

O arquivo [`projeto_credito_supervisionado.ipynb`](projeto_credito_supervisionado.ipynb) cobre todo o ciclo de vida do pipeline de Machine Learning:

1. **Setup e EDA (Exploratory Data Analysis)**:
   - Configura o Python e pacotes. Valida a infraestrutura através de virtualenvs.
   - Analisa e trata assimetrias multivariadas, correlações e a distribuição do atributo alvo (`SeriousDlqin2yrs`).
2. **Pré-Processamento Linear**:
   - Desenvolve Pipelines do `scikit` usando transformadores como `StandardScaler`.
3. **Modelagem Baseline**:
   - Treinamento e análise direcional com o classificador linear determinístico (`Perceptron`), expondo os pesos das features e mostrando a limitação das fronteiras lineares.
4. **Árvores de Decisão**:
   - Treinamentos não lineares, validando caminhos explicáveis.
   - Aplicação de `StratifiedKFold` e `GridSearchCV` controlando *max_depth* e variâncias do modelo para evitar overfitting severo.
5. **Ensambles e Otimização Final**:
   - Modelagem de complexidade superior usando agrupamento via `RandomForestClassifier`.
6. **Desfecho Analítico**:
   - Produção de um framework comparador interativo (Precision-Recall / AUC-ROC).
   - Levantamento de um racional conclusivo: os impactos de aprovar maus pagadores (Falso Negativo) vs negar bons tomadores (Falso Positivo).

### Resultados Esperados
Ao final do processo de triagem e *cross validation*, o **Random Forest** figura como o melhor detentor da métrica F1, estabilizando e diminuindo as altas variações intrínsecas ao modelo unitário (Árvore Convencional), destacando featuers financeiras chave no balanço final antes das recomendações aos gestores de carteira.

### Conclusões e Viabilidade Real
Para o cenário de BB Asset / Contexto BACEN:
- O Random Forest é recomendado para o motor de scoring de alta precisão.

- A Árvore Otimizada serve como ferramenta de auditoria para o Art. 20 da LGPD, permitindo explicar negativas de crédito de forma clara.

---

## Como Executar o Projeto Localmente

Siga o passo a passo rigoroso abaixo para replicar o ambiente perfeitamente compatível a esta versão avaliativa do modelo (utilizando **Python 3.13** e o provedor nativo **pip**):

### 1. Pré-Requisitos
Certifique-se de possuir o Python versão 3.13, e que o arquivo zip ou a interface do Kaggle CLI esteja provisionada se desejar forçar a regressão do dataset da nuvem.

### 2. Criação do Ambiente Virtual
No terminal, dentro da pasta do projeto aberto, declare as entidades atômicas definindo o nome *credito*:
```bash
# Cria o ambiente isolado
python3.13 -m venv credito

# Ativa o ambiente (macOS/Linux)
source credito/bin/activate

# Ativa o ambiente (Windows)
credito\Scripts\activate
```

### 3. Atualizando o Gerenciador e Configurando Repositórios
Garanta que a infraestrutura se comunique limpa com os repos base:
```bash
# Opcional mas recomendado: Upgrade no pip core
pip install --upgrade pip

# Instala as bibliotecas exatas e requeridas
pip install numpy pandas scikit-learn matplotlib seaborn kaggle ipykernel
```

### 4. Gerenciamento do Kernel de Reprodução Experimental (Jupyter)
Gere um manifesto congelado do projeto para pareamento com as docstrings.
```bash
# Salva as assinaturas exatas localmente
pip freeze > requirements.txt
```

### 5. Executando o Notebook
Agora que o repositório base suporta o core analítico, inicialize-o:
```bash
jupyter notebook projeto_credito_supervisionado.ipynb
```

*(Se o seu Jupyter não encontrar o abiente nativo ou global, certifique-se de registrar o ipykernel instalado previamente dentro deste env com: `python -m ipykernel install --user --name=credito`)*.

---
<div align="center">
  <small>Desenvolvido para fins acadêmicos e analíticos.<br>Março / 2026</small>
</div>
