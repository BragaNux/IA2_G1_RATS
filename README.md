# Padrões de Avistamento de Ratos em Nova York — Hotspots e Eficiência de Atendimento

**Trabalho 1 — Inteligência Artificial II (2026/02) — Faculdade Antonio Meneghetti (AMF)**

Autores: Brayan Martins, Mell Campos, Nicolas Marin

## Problema

A proliferação de roedores em centros urbanos densos como Nova York representa um
desafio de saúde pública e saneamento. A prefeitura recebe milhares de denúncias via
311, mas a distribuição espacial e temporal dessas ocorrências é heterogênea, o que
dificulta a alocação preditiva de inspeções sanitárias.

Este trabalho analisa os padrões espaciais e temporais de avistamentos de ratos em
NYC para (1) identificar hotspots geográficos de infestação via **clustering
(DBSCAN)** e (2) prever a **eficiência de atendimento** dos chamados (resolução
rápida vs. devagar) via **classificação supervisionada** (Regressão Logística e
Random Forest), apoiando a priorização de vistorias.

## Dataset

[NYC Open Data — Rat Sightings (311 Service Requests)](https://data.cityofnewyork.us/Social-Services/Rat-Sightings/3q43-55fe),
disponibilizado pelo Department of Health and Mental Hygiene (DOHMH). Uma cópia
local está em `data/Rat_Sightings.csv` (~102 mil registros, 2010–2016).

## Estrutura do repositório

```
.
├── data/
│   └── Rat_Sightings.csv          # dataset original
├── notebook/
│   └── rat_sightings_analysis.ipynb  # notebook consolidado (EDA + clustering + classificação), já executado
├── figures/                       # gráficos gerados pelos scripts/notebook
├── reports/
│   └── Trabalho1_IA2_RatSightings_Relatorio.pdf  # relatório técnico (modelo exigido pela disciplina)
├── analysis.py                    # etapa 1: carga, limpeza, feature engineering, EDA
├── clustering.py                  # etapa 2: DBSCAN (hotspots) + KMeans (comparação)
├── classification.py              # etapa 3: Regressão Logística + Random Forest
├── build_report.py                # gera o relatório PDF a partir dos resultados
└── requirements.txt
```

## Como reproduzir

```bash
pip install -r requirements.txt

# opção A — rodar os scripts em sequência
python3 analysis.py
python3 clustering.py
python3 classification.py
python3 build_report.py

# opção B — abrir o notebook consolidado
jupyter notebook notebook/rat_sightings_analysis.ipynb
```

Os scripts leem `data/Rat_Sightings.csv` e salvam figuras em `figures/` e resultados
intermediários (`.pkl`/`.json`) em `data/`. O relatório final é gerado em
`reports/`.

## Resumo dos resultados

- **EDA:** sazonalidade nítida (picos no verão), Brooklyn concentra o maior volume
  de chamados, mediana de tempo de resolução = 9 dias.
- **Clustering (DBSCAN):** 74 hotspots geográficos identificados, concentrados no
  eixo Brooklyn–Manhattan–Bronx; eps calibrado via gráfico k-distance.
- **Classificação:** AUC ≈ 0.60 (Random Forest) — desempenho modesto, sem
  overfitting relevante (gap treino/teste pequeno), indicando que bairro/tipo de
  local/sazonalidade têm poder preditivo limitado sobre o tempo de resolução.
  Discussão crítica completa no relatório (`reports/`).

## Contribuições da equipe (Pode mudar, só pra ter uma base inicial, mas o intuito é todos ajudarem em todos os processos)

- Mell - Relatório
- Brayan - Análise e Classificação
- Nicolas - Clustering

## Modelos Utilizados

- Random Forest - Brayan Martins, testes iniciais
- Gradient Boosting - Nicolas Marin, testes iniciais

## Focos principais agorar

- Consertarr o vazamento do cluster geográfico, ajuste o KMeans só com as coordenadas do treino, depois use .predict() (não .fit_predict()) nas coordenadas de teste.
- Troque o split único por validação cruzada (StratifiedKFold, 5 folds), um único split de 80/20 pode dar sorte ou azar, CV te dá uma média + desvio padrão, muito mais confiável.
- Separe tuning de avaliação final, hiperparâmetros (profundidade, n_estimators, learning_rate) devem ser escolhidos via GridSearchCV/RandomizedSearchCV dentro dos folds de treino, nunca olhando o resultado no teste e ajustando depois. Eu usei valores que chutei de cabeça, sem tuning nenhum.
- Vazamento leve a checar, month, season e day_of_week vêm todos da mesma Created Date, não é vazamento (são conhecidos no momento da abertura do chamado), mas é redundância/multicolinearidade. Não quebra nada em árvores, mas vale mencionar se for tentar modelo linear.
- Mantenha o teste intocado até o fim, nada de olhar métrica de teste, ajustar hiperparâmetro, testar de novo, isso também é uma forma (sutil) de vazamento por "espiar" o conjunto de avaliação repetidamente.

ADICIONA DEPOIS O RHAUANI AO REPOSITORIO
