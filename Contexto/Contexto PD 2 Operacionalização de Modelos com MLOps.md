Olá, Fabio.

No Projeto de Disciplina anterior, você construiu, comparou e analisou modelos supervisionados clássicos, explorando suas limitações, capacidades de generalização e trade-offs entre desempenho e complexidade. Esse percurso permitiu compreender como diferentes algoritmos se comportam diante de um mesmo problema real.

Neste Projeto de Disciplina, o foco deixa de ser o **modelo isolado** e passa a ser o **projeto de machine learning como um sistema**. Você irá atuar como um **engenheiro de machine learning**, responsável por estruturar, organizar, rastrear, comparar e operacionalizar experimentos, conectando decisões técnicas a objetivos de negócio, custo computacional e viabilidade de produção.

Este projeto representa a **evolução natural do projeto anterior**. Os experimentos já realizados não serão descartados, mas reorganizados, refinados e sistematizados dentro de uma estrutura profissional, reprodutível e orientada à operação.

## Objetivo Geral do Projeto

Estruturar um projeto completo de machine learning, integrando **planejamento, fundação de dados, experimentação, validação, redução de dimensionalidade, rastreamento de experimentos e operacionalização**, utilizando scikit-learn e MLflow, com foco em tomada de decisão técnica e impacto de negócio.

## Contexto do Projeto

Você deve assumir que:

- O problema de negócio, o dataset e os experimentos iniciais já foram explorados no Projeto de Disciplina anterior.
- Existem múltiplas abordagens de modelagem já testadas.
- O desafio agora é **organizar, sistematizar e escalar tecnicamente** esse trabalho.

O mesmo dataset deve ser reutilizado ao longo de todo o projeto, permitindo comparações consistentes entre abordagens, aplicação de técnicas de redução de dimensionalidade e análise da evolução dos modelos.

Parte 1 **Estruturação do Projeto de Machine Learning**

Nesta etapa, você deve **reorganizar o trabalho realizado anteriormente**, assumindo explicitamente a perspectiva de um engenheiro de machine learning.

O objetivo não é criar novos modelos, mas **dar forma de engenharia** ao projeto, reduzindo a dependência de notebooks exploratórios e estabelecendo uma base estruturada para experimentação e operação.

Você deve:

- Mapear os experimentos já realizados, identificando modelos testados, métricas utilizadas, principais resultados e limitações.
- Definir de forma explícita o objetivo técnico do projeto, critérios de sucesso e métricas de negócio associadas.
- Reestruturar o projeto em código, garantindo que a lógica principal de preparação, treinamento e validação esteja em scripts ou módulos reutilizáveis, e que notebooks sejam usados apenas para exploração ou visualização.
- Analisar os dados sob a ótica de engenharia, apontando riscos iniciais de qualidade, viés e generalização.

**Objetivo da etapa:** demonstrar que você compreende machine learning como um **processo de engenharia**, e não apenas como execução de algoritmos.

Parte 2 **Fundação de Dados e Diagnóstico Inicial**

Nesta etapa, você deve estabelecer a **base de dados confiável** sobre a qual todos os experimentos serão conduzidos.

Você deve:

- Estruturar a ingestão de dados, definindo fontes, formatos e estratégias de amostragem.
- Diagnosticar problemas de qualidade de dados, como valores ausentes, ruído, inconsistências e possíveis vieses.
- Analisar o impacto desses problemas na generalização, estabilidade dos resultados e risco de overfitting.
- Documentar limitações estruturais do dataset que não possam ser corrigidas apenas com modelagem.

**Objetivo da etapa:** garantir que a experimentação subsequente seja baseada em dados compreendidos, controlados e tecnicamente defensáveis.

Parte 3 **Experimentação Sistemática de Modelos**

Com a base de dados estruturada e compreendida, você deve conduzir **experimentos controlados de modelagem**.

Você deve:

- Executar experimentos comparativos entre abordagens candidatas já exploradas no projeto anterior.
- Selecionar modelos considerando desempenho preditivo, custo computacional, complexidade e interpretabilidade.
- Construir pipelines end-to-end de preparação de dados, treinamento e validação utilizando scikit-learn.
- Ajustar modelos com validação cruzada e busca de hiperparâmetros.
- Registrar todos os experimentos no MLflow, incluindo parâmetros, métricas e versões de modelos.

**Objetivo da etapa:** transformar exploração em **evidência experimental comparável**, capaz de sustentar decisões técnicas.

Parte 4 **Controle de Complexidade e Redução de Dimensionalidade**

Nesta etapa, o foco é o **controle consciente da complexidade do modelo**, do custo computacional e da generalização.

Você deve:

- Analisar a necessidade de redução de dimensionalidade com base nos resultados experimentais obtidos anteriormente.
- **Escolher e aplicar duas técnicas de redução de dimensionalidade**, dentre PCA, LDA e t-SNE, **justificando explicitamente a escolha de cada uma** em função das características dos dados e do objetivo do modelo.
- Para cada técnica escolhida: - integrar a redução de dimensionalidade ao pipeline de modelagem, - treinar novamente os classificadores.
- Comparar o desempenho dos modelos **com e sem redução de dimensionalidade**, analisando: - impacto no resultado final da classificação, - custo computacional de treinamento e inferência, - efeitos sobre a interpretabilidade do modelo.
- Discutir os trade-offs observados e justificar se a redução de dimensionalidade é ou não adequada ao contexto do problema. **Objetivo da etapa:** demonstrar domínio técnico sobre dimensionalidade, overfitting e eficiência do modelo.

Parte 5 **Consolidação Experimental e Seleção Final**

Nesta etapa, você deve consolidar os resultados experimentais e justificar tecnicamente a escolha do modelo final.

Você deve:

- Analisar comparativamente os experimentos registrados no MLflow.
- Justificar a seleção da abordagem final com base em métricas técnicas, custo computacional, complexidade e viabilidade de operação.
- Definir explicitamente o modelo candidato à operação.

**Objetivo da etapa:** fechar o ciclo experimental com uma decisão técnica clara e justificável.

Parte 6 **Operacionalização e Simulação de Produção**

Na etapa final, você deve **simular ou implementar a operação do modelo selecionado**.

Você deve:

- Persistir modelos treinados em scikit-learn de forma versionada.
- Executar inferência consistente a partir de modelos persistidos.
- Empacotar modelos como artefatos de inferência.
- Expor o modelo por meio de um serviço simples de inferência.
- Integrar o deploy do modelo a um pipeline de CI/CD simulado ou real.
- Definir métricas técnicas do modelo e métricas de impacto de negócio.
- Analisar desempenho pós-deploy.
- Detectar drift de dados e de modelo por meio de comparação estatística.
- Monitorar métricas e versões no MLflow.
- Planejar estratégias de re-treinamento e aprendizado contínuo.

**Objetivo da etapa:** demonstrar que você compreende machine learning como um **sistema vivo em produção**, sujeito a degradação, mudança de dados e necessidade de monitoramento contínuo.

## Entregáveis

Você deve entregar:

1. Um repositório organizado contendo pipelines de dados e modelos, código de experimentação e configuração do MLflow.
2. Um relatório técnico estruturado, contendo decisões de projeto, análise comparativa de experimentos e justificativa da abordagem final.
3. Uma demonstração funcional ou simulação de operação do modelo, incluindo inferência, versionamento e monitoramento. (video)

Código funcional isoladamente não é suficiente. A avaliação considera fortemente **estrutura, rastreabilidade, interpretação e decisões técnicas**.

## Considerações Finais

Este Projeto de Disciplina consolida a transição do aluno de executor de modelos para **engenheiro de machine learning**. O foco não está em maximizar métricas isoladas, mas em demonstrar visão sistêmica, maturidade técnica e responsabilidade profissional na construção e operação de sistemas de machine learning.

### Status da entrega

# Rubrica de Avaliação — Projeto de Machine Learning

|Competência|Critério de Avaliação|Não demonstrou o item de rubrica|Demonstrou o item de rubrica|
|---|---|---|---|
|1. Estruturar um projeto de machine learning, integrando planejamento, experimentação e validação ao contexto de negócio|O aluno identificou corretamente o contexto do problema, os objetivos técnicos do projeto e as métricas de negócio associadas, conectando-os de forma coerente?|||
|1. Estruturar um projeto de machine learning, integrando planejamento, experimentação e validação ao contexto de negócio|O aluno planejou o projeto de machine learning diferenciando claramente experimentação exploratória de um projeto orientado à entrega?|||
|1. Estruturar um projeto de machine learning, integrando planejamento, experimentação e validação ao contexto de negócio|O aluno estruturou o projeto em uma organização de código adequada à prática de engenharia, reduzindo a dependência de notebooks e definindo fluxos claros?|||
|1. Estruturar um projeto de machine learning, integrando planejamento, experimentação e validação ao contexto de negócio|O aluno analisou o papel do engenheiro de machine learning na tomada de decisão técnica, justificando escolhas com base em impacto prático e viabilidade?|||
|2. Projetar pipelines de dados e features para evitar overfitting e reduzir custo computacional|O aluno identificou problemas de qualidade de dados, vieses e limitações estruturais do dataset, analisando seus impactos na generalização?|||
|2. Projetar pipelines de dados e features para evitar overfitting e reduzir custo computacional|O aluno modelou pipelines de pré-processamento e preparação de dados utilizando scikit-learn, evitando vazamento de dados?|||
|2. Projetar pipelines de dados e features para evitar overfitting e reduzir custo computacional|O aluno implementou técnicas de redução de dimensionalidade escolhidas de forma justificada, integrando-as corretamente ao pipeline de modelagem?|||
|2. Projetar pipelines de dados e features para evitar overfitting e reduzir custo computacional|O aluno analisou o impacto da redução de dimensionalidade no desempenho da classificação, no custo computacional e na interpretabilidade do modelo?|||
|3. Construir experimentos reprodutíveis, integrando comparação de modelos, ajuste de hiperparâmetros e rastreamento de métricas com MLflow|O aluno planejou experimentos de machine learning de forma estruturada, definindo comparações claras entre abordagens candidatas?|||
|3. Construir experimentos reprodutíveis, integrando comparação de modelos, ajuste de hiperparâmetros e rastreamento de métricas com MLflow|O aluno implementou experimentos comparativos com validação cruzada e ajuste de hiperparâmetros de maneira consistente?|||
|3. Construir experimentos reprodutíveis, integrando comparação de modelos, ajuste de hiperparâmetros e rastreamento de métricas com MLflow|O aluno registrou corretamente parâmetros, métricas e versões de modelos no MLflow, garantindo rastreabilidade?|||
|3. Construir experimentos reprodutíveis, integrando comparação de modelos, ajuste de hiperparâmetros e rastreamento de métricas com MLflow|O aluno analisou criticamente os resultados experimentais no MLflow, justificando tecnicamente a seleção da abordagem final?|||
|4. Operacionalizar modelos de machine learning em produção com scikit-learn e MLflow|O aluno persistiu modelos treinados de forma versionada e reprodutível, garantindo consistência entre treinamento e inferência?|||
|4. Operacionalizar modelos de machine learning em produção com scikit-learn e MLflow|O aluno implementou um fluxo de inferência funcional, expondo o modelo por meio de um serviço ou interface adequada?|||
|4. Operacionalizar modelos de machine learning em produção com scikit-learn e MLflow|O aluno definiu métricas técnicas e métricas de impacto de negócio para acompanhamento do modelo em operação?|||
|4. Operacionalizar modelos de machine learning em produção com scikit-learn e MLflow|O aluno analisou aspectos de operação, incluindo monitoramento, detecção de drift e estratégias de re-treinamento, demonstrando visão de ciclo de vida do modelo?|||



