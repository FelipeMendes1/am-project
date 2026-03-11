"""
Experimentos complementares — RQs 1–8.
Reutiliza componentes de treino.py para experimentos de volume,
conjuntos de features e métodos de mitigação.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample

# Importa componentes de treino.py (também configura sys.path para fadm/kamishima)
from treino import (
    PrejudiceRemoverModel,
    carregar_e_preparar_dados,
    BASES_MAP, SENSITIVE, TARGET, FEATURES, CONT_FEATURES,
)

# ── Configurações ─────────────────────────────────────────────────────────────
SEED     = 42
N_TREINO = 10000
MAX_TESTE = 20000
OUT_DIR  = './resultados'

VOLUMES = [1000, 2500, 5000, 7500, 10000]

FEATURE_SETS = {
    # Nível 1 — contexto transacional: valor relativo ao histórico do titular
    'transacional_2':   ['pct_saldo_gasto', 'zscore_valor_no_titular'],
    # Nível 2 — + contexto temporal e tipo de lançamento
    'temporal_tipo_4':  ['pct_saldo_gasto', 'zscore_valor_no_titular',
                         'final_de_semana', 'is_cnab220'],
    # Nível 3 — + padrões de comportamento do titular
    'comportamental_6': ['pct_saldo_gasto', 'zscore_valor_no_titular',
                         'final_de_semana', 'is_cnab220',
                         'n_transacoes_dia_tit', 'prop_cnab220_tit'],
    # Nível 4 — conjunto completo com z-scores por ramo de atividade
    'completo_10':       FEATURES,
}

# ── Funções auxiliares (module-level para pickling no Windows/loky) ───────────

def _metricas(y_test, y_pred, s_test):
    mask_u = s_test == 0
    mask_p = s_test == 1
    taxa_u = (y_pred[mask_u] == 0).mean() if mask_u.any() else 0.0
    taxa_p = (y_pred[mask_p] == 0).mean() if mask_p.any() else 0.0
    spd = taxa_u - taxa_p
    di  = (taxa_u / taxa_p) if taxa_p > 0 else 0.0
    return {
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'abs_spd':   abs(spd),
        'di':        di,
    }


def _balancear(df, n_treino):
    n = min(len(df[df[TARGET] == 0]), len(df[df[TARGET] == 1]), n_treino // 2)
    return pd.concat([
        resample(df[df[TARGET] == 0], n_samples=n, replace=False, random_state=SEED),
        resample(df[df[TARGET] == 1], n_samples=n, replace=False, random_state=SEED),
    ]).sample(frac=1, random_state=SEED)


def _pesos_reweighting(s, y):
    """Pesos Kamiran & Calders (2012): w(s,y) = P(S)·P(Y) / P(S,Y)"""
    n = len(s)
    pesos = np.ones(n, dtype=np.float64)
    for sv in np.unique(s):
        for yv in np.unique(y):
            mask = (s == sv) & (y == yv)
            if mask.sum() == 0:
                continue
            p_s  = (s == sv).mean()
            p_y  = (y == yv).mean()
            p_sy = mask.mean()
            pesos[mask] = (p_s * p_y) / p_sy
    return pesos


def _carregar_teste(nome_teste):
    df_te = carregar_e_preparar_dados(nome_teste)
    if len(df_te) > MAX_TESTE:
        df_te = df_te.sample(n=MAX_TESTE, random_state=SEED)
    return df_te


def _escalar(X_tr, X_te, fs_cols):
    """StandardScaler seletivo: escala apenas features contínuas dentro do feature set."""
    cont_idx = [i for i, c in enumerate(fs_cols) if c in CONT_FEATURES]
    scaler = StandardScaler()
    X_tr_s = X_tr.copy().astype(np.float64)
    X_te_s = X_te.copy().astype(np.float64)
    if cont_idx:
        X_tr_s[:, cont_idx] = scaler.fit_transform(X_tr[:, cont_idx])
        X_te_s[:, cont_idx] = scaler.transform(X_te[:, cont_idx])
    return X_tr_s, X_te_s


# ── Experimento 1 — Volume (RQs 2, 4, 5, 6) ──────────────────────────────────
# 10 bases × 9 testes × 5 volumes = 450 linhas

def _job_volume(nome_treino):
    df_tr = carregar_e_preparar_dados(nome_treino)
    # Pré-carrega todos os dados de teste uma vez por job
    testes = {k: _carregar_teste(k) for k in BASES_MAP if k != nome_treino}
    resultados = []

    for vol in VOLUMES:
        df_bal = _balancear(df_tr, vol)
        X_tr_raw = df_bal[FEATURES].values
        y_tr     = df_bal[TARGET].values

        for nome_teste, df_te in testes.items():
            X_te_raw = df_te[FEATURES].values
            X_tr_s, X_te_s = _escalar(X_tr_raw, X_te_raw, FEATURES)

            lr = LogisticRegression(max_iter=1000, random_state=SEED)
            lr.fit(X_tr_s, y_tr)
            y_pred = lr.predict(X_te_s)

            m = _metricas(df_te[TARGET].values, y_pred, df_te[SENSITIVE].values)
            m.update({
                'base_treino':  nome_treino,
                'serie_treino': BASES_MAP[nome_treino][1],
                'vies_treino':  BASES_MAP[nome_treino][2],
                'volume':       vol,
            })
            resultados.append(m)

    return resultados


# ── Experimento 2 — Feature sets (RQs 1, 3) ──────────────────────────────────
# 10 bases × 9 testes × 4 feature sets = 360 linhas

def _job_features(nome_treino):
    df_tr = carregar_e_preparar_dados(nome_treino)
    df_bal = _balancear(df_tr, N_TREINO)
    testes = {k: _carregar_teste(k) for k in BASES_MAP if k != nome_treino}
    resultados = []

    for fs_name, fs_cols in FEATURE_SETS.items():
        X_tr_raw = df_bal[fs_cols].values
        y_tr     = df_bal[TARGET].values

        for nome_teste, df_te in testes.items():
            X_te_raw = df_te[fs_cols].values
            X_tr_s, X_te_s = _escalar(X_tr_raw, X_te_raw, fs_cols)

            lr = LogisticRegression(max_iter=1000, random_state=SEED)
            lr.fit(X_tr_s, y_tr)
            y_pred = lr.predict(X_te_s)

            m = _metricas(df_te[TARGET].values, y_pred, df_te[SENSITIVE].values)
            m.update({
                'base_treino':  nome_treino,
                'serie_treino': BASES_MAP[nome_treino][1],
                'vies_treino':  BASES_MAP[nome_treino][2],
                'feature_set':  fs_name,
                'n_features':   len(fs_cols),
            })
            resultados.append(m)

    return resultados


# ── Experimento 3 — Métodos de mitigação (RQs 7, 8) ──────────────────────────
# 10 bases × 9 testes × 4 métodos = 360 linhas

def _job_mitigacao(nome_treino):
    df_tr  = carregar_e_preparar_dados(nome_treino)
    testes = {k: _carregar_teste(k) for k in BASES_MAP if k != nome_treino}
    cont_idx = [i for i, c in enumerate(FEATURES) if c in CONT_FEATURES]
    resultados = []

    # ── Scaler compartilhado (ajustado nos dados balanceados) ────────────────
    df_bal  = _balancear(df_tr, N_TREINO)
    X_bal   = df_bal[FEATURES].values.astype(np.float64)
    y_bal   = df_bal[TARGET].values
    s_bal   = df_bal[SENSITIVE].values

    scaler_shared = StandardScaler()
    X_bal_s = X_bal.copy()
    if cont_idx:
        X_bal_s[:, cont_idx] = scaler_shared.fit_transform(X_bal[:, cont_idx])

    # 1) LR sem mitigação — amostra bruta escalada pelo mesmo scaler
    df_bruto  = df_tr.sample(n=min(len(df_tr), N_TREINO), random_state=SEED)
    X_bruto   = df_bruto[FEATURES].values.astype(np.float64)
    X_bruto_s = X_bruto.copy()
    if cont_idx:
        X_bruto_s[:, cont_idx] = scaler_shared.transform(X_bruto[:, cont_idx])
    lr_bruto = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_bruto.fit(X_bruto_s, df_bruto[TARGET].values)

    # 2) LR balanceado
    lr_bal = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_bal.fit(X_bal_s, y_bal)

    # 3) LR reponderado — pesos Kamiran & Calders (2012)
    w_bal = _pesos_reweighting(s_bal, y_bal)
    lr_rw = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_rw.fit(X_bal_s, y_bal, sample_weight=w_bal)

    # 4) Prejudice Remover η=10 — com SENSITIVE como última coluna
    fs_pr    = FEATURES + [SENSITIVE]
    cont_pr  = [i for i, c in enumerate(fs_pr) if c in CONT_FEATURES]
    scaler_pr = StandardScaler()
    X_pr_raw = df_bal[fs_pr].values.astype(np.float64)
    X_pr_s   = X_pr_raw.copy()
    if cont_pr:
        X_pr_s[:, cont_pr] = scaler_pr.fit_transform(X_pr_raw[:, cont_pr])
    pr_model = PrejudiceRemoverModel(eta=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr_model.fit(X_pr_s, y_bal)

    for nome_teste, df_te in testes.items():
        y_te    = df_te[TARGET].values
        s_te    = df_te[SENSITIVE].values
        X_te    = df_te[FEATURES].values.astype(np.float64)
        X_te_s  = X_te.copy()
        if cont_idx:
            X_te_s[:, cont_idx] = scaler_shared.transform(X_te[:, cont_idx])

        X_te_pr_raw = df_te[fs_pr].values.astype(np.float64)
        X_te_pr_s   = X_te_pr_raw.copy()
        if cont_pr:
            X_te_pr_s[:, cont_pr] = scaler_pr.transform(X_te_pr_raw[:, cont_pr])

        base_info = {
            'base_treino':  nome_treino,
            'serie_treino': BASES_MAP[nome_treino][1],
            'vies_treino':  BASES_MAP[nome_treino][2],
        }

        for metodo, y_pred in [
            ('lr_sem_mitigacao',        lr_bruto.predict(X_te_s)),
            ('lr_balanceado',           lr_bal.predict(X_te_s)),
            ('lr_reponderado',          lr_rw.predict(X_te_s)),
            ('prejudice_remover_eta10', pr_model.predict(X_te_pr_s)),
        ]:
            m = _metricas(y_te, y_pred, s_te)
            m.update({'metodo': metodo, **base_info})
            resultados.append(m)

    return resultados


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("Experimento 1 — Volume de dados (RQs 2, 4, 5, 6)...")
    jobs1 = Parallel(n_jobs=-1)(delayed(_job_volume)(k) for k in BASES_MAP)
    df_vol = pd.DataFrame([r for sub in jobs1 for r in sub])
    df_vol.to_csv(os.path.join(OUT_DIR, 'exp_volume.csv'), index=False)
    print(f"  -> {len(df_vol)} linhas  (esperado: 450)  salvas em exp_volume.csv")

    print("Experimento 2 — Conjuntos de features (RQs 1, 3)...")
    jobs2 = Parallel(n_jobs=-1)(delayed(_job_features)(k) for k in BASES_MAP)
    df_feat = pd.DataFrame([r for sub in jobs2 for r in sub])
    df_feat.to_csv(os.path.join(OUT_DIR, 'exp_features.csv'), index=False)
    print(f"  -> {len(df_feat)} linhas  (esperado: 360)  salvas em exp_features.csv")

    print("Experimento 3 — Métodos de mitigação (RQs 7, 8)...")
    jobs3 = Parallel(n_jobs=-1)(delayed(_job_mitigacao)(k) for k in BASES_MAP)
    df_mit = pd.DataFrame([r for sub in jobs3 for r in sub])
    df_mit.to_csv(os.path.join(OUT_DIR, 'exp_mitigacao.csv'), index=False)
    print(f"  -> {len(df_mit)} linhas  (esperado: 360)  salvas em exp_mitigacao.csv")

    print(f"\nConcluído em {time.time() - t0:.1f} segundos.")


if __name__ == '__main__':
    main()
