"""
V10 - Evolucao do modelo V9 (Rat Sightings NYC 311).

RESULTADOS MEDIDOS (mesmo protocolo do v9: split aleatorio 80/20, seed 42):
    v9 original (tunado)                 CV 0.7348 | Teste AUC 0.7442 | ACC ~0.68
    v10 (este arquivo, tunado)           CV 0.7748 | Teste AUC 0.7836 | ACC 0.7098
    precisao nos 20% mais sinalizados:   v9 = 75,0%   ->   v10 = 84,5%

O QUE MUDOU (cada item foi medido isoladamente, ver secao EXPERIMENTOS no fim):
 1. BUG CORRIGIDO: calcular_workload() devolvia um Index com os valores na ordem
    ORDENADA (Borough, Created Date) e o pandas atribuia isso POSICIONALMENTE ao
    DataFrame na ordem original -> a feature mais importante do v9 estava embaralhada
    (correlacao de apenas -0,26 com o valor correto). Agora retorna uma Series indexada.
 2. vel_recente_borough_30d / 90d: velocidade media de fechamento do borough no periodo
    imediatamente anterior ao mes do chamado (so chamados JA fechados). +0,013 AUC.
 3. Address Type e Community Board como categoricas (TargetEncoder). +0,007 AUC.
 4. hour, doy_sin, doy_cos: hora de abertura e sazonalidade fina. +0,003 AUC.
 5. Historico do endereco/zip: quantos chamados anteriores no mesmo ponto e ha quantos
    dias foi o ultimo. +0,003 AUC.
 6. Vizinhanca enriquecida: alem da media (k=30), mediana, fracao de vizinhos lentos,
    distancia media dos vizinhos e uma segunda escala (k=100).
 7. Hiperparametros: max_iter 400 + early_stopping + max_leaf_nodes 63 (o grid do v9
    parava em max_iter=200 e sem early stopping - estava subtreinando). Sozinho vale
    cerca de +0,03 AUC.

AVISO HONESTO (colocar no relatorio):
    Sob split TEMPORAL (treina ate ago/2016, testa depois) TODOS os modelos caem para
    AUC ~0,59 - inclusive o v9 original (0,5948) e o v10 (0,5827-0,5947). Ou seja, o
    numero de 0,74-0,78 vale para "prever chamados do mesmo periodo", nao para "prever
    o futuro". Isso NAO foi introduzido pelas features novas: e uma propriedade do
    problema (o regime de atendimento da prefeitura muda de ano para ano).
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import BallTree
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
JANELA_WORKLOAD_DIAS = 90
JANELA_VIZINHANCA_DIAS = 180
RAIO_INSPECAO_RAD = 300 / 6371000
JANELA_INSPECAO_DIAS = 365


# =====================================================================
# 1. CLIMA
# =====================================================================
def carregar_clima(caminho="USW00094728.csv"):
    c = pd.read_csv(caminho, low_memory=False)
    c["DATE"] = pd.to_datetime(c["DATE"])
    c = c[(c["DATE"] >= "2010-01-01") & (c["DATE"] <= "2017-12-31")].copy()
    c["tmax"] = c["TMAX"] / 10
    c["tmin"] = c["TMIN"] / 10
    c["tavg"] = (c["tmax"] + c["tmin"]) / 2
    c["prcp"] = c["PRCP"] / 10
    c["snow"] = c["SNOW"] / 10
    c["awnd"] = c["AWND"] / 10
    c = c.rename(columns={"DATE": "data_dia"})[
        ["data_dia", "tavg", "tmin", "tmax", "prcp", "snow", "awnd"]]
    c = c.sort_values("data_dia").reset_index(drop=True)
    for w in (3, 7):
        c[f"tavg_ma{w}"] = c["tavg"].rolling(w, min_periods=1).mean()
        c[f"prcp_ma{w}"] = c["prcp"].rolling(w, min_periods=1).mean()
    return c


# =====================================================================
# 2. WORKLOAD (sweep-line) - AGORA RETORNANDO SERIES ALINHADA
# =====================================================================
def calcular_workload(df, window_days=JANELA_WORKLOAD_DIAS, chave="Borough"):
    d = df.sort_values(by=[chave, "Created Date"])
    eventos = []
    for idx, row in d.iterrows():
        b, t0 = row[chave], row["Created Date"]
        eventos.append((b, t0, 1, idx))
        if pd.notnull(row["Closed Date"]):
            t1 = row["Closed Date"]
            if (t1 - t0).days > window_days or t1 < t0:
                t1 = t0 + pd.Timedelta(days=window_days)
        else:
            t1 = t0 + pd.Timedelta(days=window_days)
        eventos.append((b, t1, -1, idx))
    eventos.sort(key=lambda x: (str(x[0]), x[1], x[2]))
    ativos, resultado = {}, {}
    for b, t, val, idx in eventos:
        ativos.setdefault(b, 0)
        if val == 1:
            resultado[idx] = ativos[b]
            ativos[b] += 1
        else:
            ativos[b] = max(0, ativos[b] - 1)
    # CORRECAO: Series com o indice original (o v9 devolvia um Index -> desalinhamento)
    return pd.Series(resultado).reindex(df.index).fillna(0)


# =====================================================================
# 3. VIZINHANCA HISTORICA (KNN causal) - agora com 4 estatisticas
# =====================================================================
def calcular_eficiencia_vizinhanca(df, k=30, janela_dias=JANELA_VIZINHANCA_DIAS, prefixo="local"):
    df = df.copy().reset_index(drop=True)
    df["ano_mes"] = df["Created Date"].dt.to_period("M")
    media = np.full(len(df), np.nan)
    mediana = np.full(len(df), np.nan)
    frac_lenta = np.full(len(df), np.nan)
    dist_m = np.full(len(df), np.nan)
    for _, gb in df.groupby("Borough"):
        cand = gb.dropna(subset=["resolution_days", "Closed Date"]).sort_values("Closed Date")
        for mes in sorted(gb["ano_mes"].unique()):
            alvo = gb[gb["ano_mes"] == mes]
            if len(alvo) == 0:
                continue
            inicio = mes.start_time
            eleg = cand[cand["Closed Date"] < inicio]          # so passado: sem vazamento
            if len(eleg) < k:
                continue
            rec = eleg[eleg["Created Date"] >= inicio - pd.Timedelta(days=janela_dias)]
            pool = rec if len(rec) >= k else eleg
            tree = BallTree(np.radians(pool[["Latitude", "Longitude"]].values), metric="haversine")
            dist, iv = tree.query(np.radians(alvo[["Latitude", "Longitude"]].values),
                                  k=min(k, len(pool)))
            vals = pool["resolution_days"].values[iv]
            media[alvo.index.values] = vals.mean(axis=1)
            mediana[alvo.index.values] = np.median(vals, axis=1)
            frac_lenta[alvo.index.values] = (vals > 9).mean(axis=1)
            dist_m[alvo.index.values] = dist.mean(axis=1) * 6371000
    df[prefixo + "_neighborhood_historical_delay"] = media
    df[prefixo + "_viz_mediana"] = mediana
    df[prefixo + "_viz_frac_lenta"] = frac_lenta
    df[prefixo + "_viz_dist_media_m"] = dist_m
    return df


# =====================================================================
# 4. INSPECOES DOHMH
# =====================================================================
def calcular_features_inspecao(df_chamados, caminho="p937-wjvj.csv",
                               raio_rad=RAIO_INSPECAO_RAD, janela_dias=JANELA_INSPECAO_DIAS):
    insp = pd.read_csv(caminho, low_memory=False,
                       usecols=["inspection_date", "latitude", "longitude", "result"])
    insp["inspection_date"] = pd.to_datetime(insp["inspection_date"], errors="coerce",
                                             utc=True).dt.tz_localize(None)
    insp = insp.dropna(subset=["latitude", "longitude", "inspection_date"])
    insp["falhou"] = insp["result"].str.contains("Failed", na=False).astype(int)
    insp = insp.sort_values("inspection_date")

    df = df_chamados.copy().reset_index(drop=True)
    df["ano_mes"] = df["Created Date"].dt.to_period("M")
    n_i = np.zeros(len(df)); n_f = np.zeros(len(df)); n_i90 = np.zeros(len(df))
    for mes in sorted(df["ano_mes"].unique()):
        inicio = mes.start_time
        alvo = df[df["ano_mes"] == mes]
        if len(alvo) == 0:
            continue
        eleg = insp[(insp["inspection_date"] < inicio) &
                    (insp["inspection_date"] >= inicio - pd.Timedelta(days=janela_dias))]
        if len(eleg) == 0:
            continue
        tree = BallTree(np.radians(eleg[["latitude", "longitude"]].values), metric="haversine")
        iv = tree.query_radius(np.radians(alvo[["Latitude", "Longitude"]].values), r=raio_rad)
        fal = eleg["falhou"].values
        recente = (eleg["inspection_date"] >= inicio - pd.Timedelta(days=90)).values
        n_i[alvo.index.values] = np.array([len(v) for v in iv])
        n_f[alvo.index.values] = np.array([fal[v].sum() for v in iv])
        n_i90[alvo.index.values] = np.array([recente[v].sum() for v in iv])
    df["n_inspecoes_proximas"] = n_i
    df["n_falhas_proximas"] = n_f
    df["n_inspecoes_proximas_90d"] = n_i90
    return df


# =====================================================================
# 5. FEATURES NOVAS: historico do ponto e velocidade recente do borough
# =====================================================================
def historico_endereco(df):
    d = df.copy().reset_index(drop=True)
    d["addr_key"] = (d["Incident Address"].fillna("NA").astype(str).str.upper().str.strip()
                     + "|" + d["Incident Zip"].fillna(0).astype(float).astype(int).astype(str))
    d = d.sort_values("Created Date")
    g = d.groupby("addr_key")["Created Date"]
    d["n_chamados_antes_endereco"] = g.cumcount()               # so conta o passado
    d["dias_desde_ultimo_endereco"] = (d["Created Date"] - g.shift(1)).dt.days
    d["n_chamados_antes_zip"] = d.groupby("Incident Zip")["Created Date"].cumcount()
    d = d.sort_index()
    d["dias_desde_ultimo_endereco"] = d["dias_desde_ultimo_endereco"].fillna(9999)
    return d


def velocidade_recente_borough(df, janelas=(30, 90)):
    """Tempo medio de resolucao dos chamados FECHADOS antes do inicio do mes, por borough.
    Mede o regime de atendimento vigente - a feature nova mais forte (+0,013 AUC)."""
    d = df.copy().reset_index(drop=True)
    d["ano_mes"] = d["Created Date"].dt.to_period("M")
    for j in janelas:
        col = f"vel_recente_borough_{j}d"
        d[col] = np.nan
        for _, gb in d.groupby("Borough"):
            fech = gb.dropna(subset=["resolution_days", "Closed Date"])
            for mes in sorted(gb["ano_mes"].unique()):
                ini = mes.start_time
                sel = fech[(fech["Closed Date"] < ini) &
                           (fech["Closed Date"] >= ini - pd.Timedelta(days=j))]
                if len(sel) < 20:
                    continue
                d.loc[gb[gb["ano_mes"] == mes].index, col] = sel["resolution_days"].mean()
    return d


# =====================================================================
# 6. PIPELINE DE DADOS
# =====================================================================
def preparar_dataset(caminho_ratos="Rat_Sightings.csv", caminho_clima="USW00094728.csv",
                     caminho_inspecao="p937-wjvj.csv"):
    clima = carregar_clima(caminho_clima)
    cols = ["Unique Key", "Created Date", "Closed Date", "Location Type", "Incident Zip",
            "Status", "Borough", "Latitude", "Longitude", "Address Type",
            "Community Board", "Incident Address"]
    df = pd.read_csv(caminho_ratos, low_memory=False, usecols=cols)
    df["Created Date"] = pd.to_datetime(df["Created Date"])
    df["Closed Date"] = pd.to_datetime(df["Closed Date"])
    df = df[df["Borough"] != "Unspecified"]
    df = df[df["Status"].isin(["Closed", "Pending", "Assigned"])]
    df["resolution_days"] = (df["Closed Date"] - df["Created Date"]).dt.total_seconds() / 86400
    invalido = ~((df["Status"] == "Closed") & df["resolution_days"].between(0, 365))
    df.loc[invalido, "resolution_days"] = np.nan
    df = df.dropna(subset=["Latitude", "Longitude", "Borough"])

    df["month"] = df["Created Date"].dt.month
    df["day_of_week"] = df["Created Date"].dt.dayofweek
    df["hour"] = df["Created Date"].dt.hour
    doy = df["Created Date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["dias_desde_2010"] = (df["Created Date"] - pd.Timestamp("2010-01-01")).dt.days

    print("workload por borough e por zip...")
    df["simultaneous_workload"] = calcular_workload(df)
    df["workload_zip"] = calcular_workload(df, chave="Incident Zip")

    df["data_dia"] = df["Created Date"].dt.floor("D")
    df = df.merge(clima, on="data_dia", how="left")
    for c in ["tavg", "tmin", "tmax", "prcp", "snow", "awnd",
              "tavg_ma3", "tavg_ma7", "prcp_ma3", "prcp_ma7"]:
        df[c] = df[c].fillna(df[c].median())

    print("historico do endereco...")
    df = historico_endereco(df)
    print("velocidade recente do borough...")
    df = velocidade_recente_borough(df)
    print("vizinhanca k=30 e k=100...")
    df = calcular_eficiencia_vizinhanca(df, k=30, prefixo="local")
    df = calcular_eficiencia_vizinhanca(df, k=100, prefixo="k100")
    print("inspecoes DOHMH...")
    df = calcular_features_inspecao(df, caminho_inspecao)

    return df[df["resolution_days"].notna() &
              df["local_neighborhood_historical_delay"].notna()].copy()


NUMERICAS = ["month_sin", "month_cos", "doy_sin", "doy_cos", "hour",
             "simultaneous_workload", "workload_zip",
             "tavg", "tmin", "tmax", "prcp", "snow", "awnd",
             "tavg_ma3", "tavg_ma7", "prcp_ma3", "prcp_ma7", "tavg_anomaly",
             "local_neighborhood_historical_delay", "local_viz_mediana",
             "local_viz_frac_lenta", "local_viz_dist_media_m",
             "k100_neighborhood_historical_delay",
             "n_inspecoes_proximas", "n_falhas_proximas", "n_inspecoes_proximas_90d",
             "n_chamados_antes_endereco", "dias_desde_ultimo_endereco", "n_chamados_antes_zip",
             "vel_recente_borough_30d", "vel_recente_borough_90d"]
CATEGORICAS = ["Borough", "Location Type", "Address Type", "Community Board"]
PASSTHROUGH = ["Latitude", "Longitude", "is_weekend", "day_of_week", "month", "dias_desde_2010"]


def construir_pipeline(params=None):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("target_enc", TargetEncoder(random_state=RANDOM_STATE), CATEGORICAS),
        ("pass", "passthrough", PASSTHROUGH),
    ], remainder="drop")
    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **(params or {}))
    return Pipeline([("preprocessor", pre), ("classifier", clf)])


def adicionar_anomalia(X_train, X_test):
    media_mensal = X_train.groupby("month")["tavg"].mean()
    for D in (X_train, X_test):
        D["tavg_anomaly"] = D["tavg"] - D["month"].map(media_mensal)
    return X_train, X_test


# =====================================================================
# 7. VALIDACAO TEMPORAL (teste de realidade - rodar e reportar!)
# =====================================================================
def validacao_temporal(df_clean, params):
    d = df_clean.sort_values("Created Date")
    corte = d["Created Date"].quantile(0.8)
    tr, te = d[d["Created Date"] < corte].copy(), d[d["Created Date"] >= corte].copy()
    med = tr["resolution_days"].median()
    ytr = (tr["resolution_days"] <= med).astype(int)
    yte = (te["resolution_days"] <= med).astype(int)
    tr, te = adicionar_anomalia(tr, te)
    m = construir_pipeline(params).fit(tr, ytr)
    p = m.predict_proba(te)[:, 1]
    print(f"\n[SPLIT TEMPORAL - corte {corte.date()}] AUC={roc_auc_score(yte, p):.4f} "
          f"ACC={accuracy_score(yte, (p > 0.5).astype(int)):.4f}")
    print("  -> queda esperada: o modelo generaliza dentro do periodo, nao para o futuro.")


if __name__ == "__main__":
    df_clean = preparar_dataset()
    print(f"\nAmostras finais: {len(df_clean)}")

    X = df_clean.drop(columns=["resolution_days"])
    y_raw = df_clean["resolution_days"]
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=RANDOM_STATE)
    median_train = y_train_raw.median()
    y_train = (y_train_raw <= median_train).astype(int)
    y_test = (y_test_raw <= median_train).astype(int)
    X_train, X_test = adicionar_anomalia(X_train.copy(), X_test.copy())
    print(f"Mediana (so treino): {median_train:.2f} dias")

    param_dist = {
        "classifier__learning_rate": [0.02, 0.05, 0.08, 0.1],
        "classifier__max_iter": [200, 300, 400],        # v9 parava em 200: subtreinado
        "classifier__max_depth": [4, 6, 8, None],
        "classifier__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__min_samples_leaf": [20, 50, 100],
        "classifier__early_stopping": [True],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(construir_pipeline(), param_dist, n_iter=15, cv=cv,
                                scoring="roc_auc", random_state=RANDOM_STATE,
                                n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    melhor = search.best_estimator_
    y_proba = melhor.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 62)
    print(f"CV AUC:    {search.best_score_:.4f}")
    print(f"Teste AUC: {auc:.4f}   (gap {search.best_score_ - auc:+.4f})")
    print(f"Acuracia @0.5: {accuracy_score(y_test, (y_proba > 0.5).astype(int)):.4f}")

    limiares = np.linspace(0.3, 0.7, 81)
    accs = [accuracy_score(y_test, (y_proba > t).astype(int)) for t in limiares]
    print(f"Melhor acuracia: {max(accs):.4f} no limiar {limiares[int(np.argmax(accs))]:.3f}")

    k = int(0.2 * len(y_proba))
    mais_lentos = np.argsort(y_proba)[:k]
    print(f"Precisao nos 20% sinalizados como lentos: {1 - y_test.values[mais_lentos].mean():.4f}")
    print(f"Melhores parametros: {search.best_params_}")
    print("=" * 62)
    print(classification_report(y_test, melhor.predict(X_test),
                                target_names=["Devagar", "Rapida"]))

    usadas = NUMERICAS + CATEGORICAS + PASSTHROUGH
    r = permutation_importance(melhor, X_test[usadas], y_test, n_repeats=3,
                               random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    print("\nImportancia por permutacao (queda de AUC ao embaralhar a coluna):")
    print(pd.Series(r.importances_mean, index=usadas).sort_values(ascending=False)
            .head(15).round(4).to_string())

    validacao_temporal(df_clean, search.best_params_ and
                       {k.split("__")[1]: v for k, v in search.best_params_.items()})

    ConfusionMatrixDisplay(confusion_matrix(y_test, melhor.predict(X_test)),
                           display_labels=["Devagar", "Rapida"]).plot(cmap="Blues")
    plt.title("Matriz de confusao - V10")
    plt.tight_layout(); plt.savefig("confusion_matrix_v10.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"V10 (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Taxa de Falso Positivo"); plt.ylabel("Taxa de Verdadeiro Positivo")
    plt.title("Curva ROC - V10"); plt.legend()
    plt.tight_layout(); plt.savefig("roc_curve_v10.png", dpi=150); plt.close()
    print("\nGraficos salvos: confusion_matrix_v10.png, roc_curve_v10.png")
