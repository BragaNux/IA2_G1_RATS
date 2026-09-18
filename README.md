# Padrões de Avistamento de Ratos em Nova York — Priorização de Atendimento (311)

**Trabalho 1, Inteligência Artificial II (2026/02), Faculdade Antonio Meneghetti (AMF)**

Autores: Mell Campos, Brayan Martins, Nicolas Marin

## Problema

A proliferação de roedores em centros urbanos densos como Nova York representa um desafio crítico de saúde pública, saneamento e qualidade de vida. Embora a prefeitura receba milhares de denúncias via 311, a distribuição espacial e temporal dessas ocorrências é muito heterogênea, o que impede uma alocação preditiva e eficiente de inspeções sanitárias. O tempo de resolução dos chamados é, em si, uma variável determinante na contenção de problemas de saúde pública e na eficiência operacional da prefeitura.

Este trabalho desenvolve um modelo preditivo capaz de classificar a resolução de chamados de avistamento em **"Rápida"** ou **"Lenta"** (classificação binária supervisionada), apoiando a priorização de vistorias pelo DOHMH.

## Datasets

1. **NYC Rat Sightings (311 Service Requests)** — [Kaggle](https://www.kaggle.com/datasets/new-york-city/nyc-rat-sightings). Registros de reclamações/avistamentos de roedores feitos por cidadãos via 311, com data de abertura/fechamento, localização geográfica, Borough, tipo de local e status do chamado. ~102 mil registros brutos (2010–2017).
2. **DOHMH Rodent Inspection** — [NYC Open Data](https://data.cityofnewyork.us/Health/Rodent-Inspection/p937-wjvj). Inspeções sanitárias/controle de pragas realizadas pelo Department of Health and Mental Hygiene, usadas para medir densidade de atividade de inspeção próxima a cada chamado. PARA DOWNLOAD DO CSV, CLIQUE NO LINK: https://data.cityofnewyork.us/resource/p937-wjvj.csv?$limit=500000&$where=inspection_date%20between%20%272010-01-01%27%20and%20%272017-12-31%27
3. **NOAA GHCN-Daily, estação Central Park (USW00094728)** — [NOAA](https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily). Temperatura mínima/máxima e precipitação diárias, usadas para construir o sinal climático (médias móveis de 3/7 dias).

Cópias locais esperadas em `data/` (ver seção de reprodução).

## Estrutura do repositório

```
.
├── data/
│ ├── Rat_Sightings_clean.csv # dataset 1, já limpo
│ ├── rodent_inspection.csv # dataset 2 (DOHMH)
│ └── noaa_central_park.csv # dataset 3 (clima)
├── modelo_v10.py # script autocontido: limpeza, features, pipeline, tuning, avaliação
├── figures/ # matriz de confusão, curva ROC, gráficos gerados pelo script
├── reports/
│ └── Trabalho1_IA2_RatSightings_Relatorio.pdf # relatório técnico (modelo exigido pela disciplina)
└── requirements.txt
```

> Nota: o projeto passou por uma fase inicial baseada em notebook + scripts separados (`analysis.py`, `clustering.py`, `classification.py`) e clustering geográfico via DBSCAN/KMeans. Essa abordagem foi descontinuada, e a versão atual (V10) consolida tudo em `modelo_v10.py`, sem etapa de clustering.

## Como reproduzir

```bash
pip install -r requirements.txt
python3 modelo_v10.py
```

O script espera os três CSVs na mesma pasta (`data/`). A preparação dos dados leva ~3 minutos; a busca de hiperparâmetros (`RandomizedSearchCV`, `n_iter=15`) leva ~5 minutos com `n_jobs=-1`. Ao final, o script imprime AUC de validação cruzada, AUC de teste, acurácia no limiar ótimo, precisão no top-20% de chamados sinalizados como lentos, importância por permutação e os resultados da validação temporal (`validacao_temporal()`), além de salvar os gráficos de avaliação em `figures/`.

## Resumo técnico

- **Alvo:** `resolution_days = Closed Date - Created Date`, binarizado contra a mediana do conjunto de treino (~9 dias).
- **Features principais:** carga de trabalho simultânea por Borough (sweep-line, janela de 90 dias), eficiência histórica de vizinhança via KNN (BallTree/haversine, k=30), densidade de inspeções DOHMH em raio de 300m, sinal climático com médias móveis, velocidade recente de atendimento por Borough (30/90 dias), além de variáveis categóricas (Address Type, Community Board) e temporais cíclicas.
- **Modelo:** `HistGradientBoostingClassifier` (scikit-learn), escolhido por aceitar Latitude/Longitude crus e não exigir agrupamento geográfico prévio.
- **Blindagem contra vazamento:** mediana do alvo calculada só no treino; features de vizinhança/inspeção usam apenas `Closed Date` anterior ao início do mês avaliado; `TargetEncoder` com cross-fitting; `remainder='drop'` no `ColumnTransformer`.
- **Avaliação:** split 80/20 (seed 42) para o resultado principal, complementado por uma validação com **split temporal** (treino até ago/2016, teste no período seguinte) para checar generalização real.

## Resultados (V10)

| Métrica | Valor |
|---|---|
| AUC (validação cruzada) | 0,7748 |
| AUC (teste, split aleatório) | 0,7836 |
| Acurácia (limiar ótimo) | 0,7078 |
| Precisão no top-20% "lentos" | 84,5% |
| AUC (split temporal) | 0,58–0,64 |

A evolução completa (V1→V10) e a discussão crítica dos resultados, incluindo a diferença entre split aleatório e split temporal, estão detalhadas no relatório em `reports/`.
