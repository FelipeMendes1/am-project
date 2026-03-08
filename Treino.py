#!/usr/bin/env python3
"""
Tema 2.2 — Prejudice Remover: equilíbrio entre fairness e acurácia
quando o nível de viés nas bases mudam.
10 ciclos: treina em 1 base, testa nas 9 restantes.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import resample
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import PrejudiceRemover
warnings.filterwarnings('ignore')

#Config
ETAS = [0.0, 5.0, 25.0, 50.0]
N_TREINO = 3000
MAX_TESTE = 12000
FEAT = ['VALOR_TRANSACAO', 'VALOR_SALDO', 'valor_total_dia',
        'n_transacoes_dia', 'valor_medio_dia', 'n_dias_atividade']
SENS = 'RAMO_ATIVIDADE_1'
LABEL = 'I-d'
SEED = 42
OUT = './resultados'
DATA = './Equidade - bases de dados públicas'

BASES = {
    'pd_v0':    ('pd_v0.csv',               'PD', 0.00),
    'pd_v0.25': ('pd_v0_25.csv',            'PD', 0.25),
    'pd_v0.5':  ('pd_v0_5.csv',             'PD', 0.50),
    'pd_v0.75': ('pd_v0_75.csv',            'PD', 0.75),
    'pd_v1':    ('pd_v1.csv',               'PD', 1.00),
    'gs_v0':    ('gs_v0_modificado(1).csv',  'GS', 0.00),
    'gs_v0.25': ('gs_v0_25_modificado.csv',  'GS', 0.25),
    'gs_v0.5':  ('gs_v0_5_modificado.csv',   'GS', 0.50),
    'gs_v0.75': ('gs_v0_75_modificado.csv',  'GS', 0.75),
    'gs_v1':    ('gs_v1_modificado.csv',     'GS', 1.00),
}


def carregar(arq):
    df = pd.read_csv(os.path.join(DATA, arq), low_memory=False)
    df = df[df[SENS].isin([1, 4])].copy()
    df[SENS] = df[SENS].replace({4: 0})
    df['DATA_LANCAMENTO'] = pd.to_datetime(df['DATA_LANCAMENTO'])
    df = df.sort_values(['CPF_CNPJ_TITULAR', 'DATA_LANCAMENTO'])
    df['data_dia'] = df['DATA_LANCAMENTO'].dt.date
    grp = df.groupby(['CPF_CNPJ_TITULAR', 'data_dia'])['VALOR_TRANSACAO']
    df['valor_total_dia'] = grp.transform('sum')
    df['n_transacoes_dia'] = grp.transform('count')
    df['valor_medio_dia'] = grp.transform('mean')
    # n dias distintos de atividade por titular (no periodo inteiro)
    df['n_dias_atividade'] = df.groupby('CPF_CNPJ_TITULAR')['data_dia'].transform('nunique')
    for f in FEAT:
        df[f] = df[f].fillna(0)
    return df


def balancear(df):
    n = min(len(df[df[LABEL]==0]), len(df[df[LABEL]==1]), N_TREINO // 2)
    if n == 0:
        return df.sample(n=min(N_TREINO, len(df)), random_state=SEED)
    c0 = resample(df[df[LABEL]==0], replace=False, n_samples=n, random_state=SEED)
    c1 = resample(df[df[LABEL]==1], replace=False, n_samples=n, random_state=SEED)
    return pd.concat([c0, c1]).sample(frac=1, random_state=SEED)


def to_bld(df, scaler=None, fit=False):
    d = df[FEAT + [SENS, LABEL]].copy().reset_index(drop=True)
    if fit:
        scaler = MinMaxScaler()
        d[FEAT] = scaler.fit_transform(d[FEAT])
    else:
        d[FEAT] = scaler.transform(d[FEAT])
    ds = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                            df=d, label_names=[LABEL],
                            protected_attribute_names=[SENS])
    return ds, scaler


def calc_metricas(y_true, y_pred, s_true):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    m_u, m_p = (s_true == 0), (s_true == 1)
    if m_u.sum() == 0 or m_p.sum() == 0:
        return dict(f1=f1, acc=acc, spd=np.nan, abs_spd=np.nan, fr=np.nan)
    r_u, r_p = y_pred[m_u].mean(), y_pred[m_p].mean()
    spd = r_u - r_p
    di = (r_u / r_p) if r_p > 0 else np.inf
    fr = min(di, 1/di) if (np.isfinite(di) and di > 0) else 0.0
    return dict(f1=f1, acc=acc, spd=spd, abs_spd=abs(spd), fr=fr)


def classificar_base(df):
    """Retorna o tipo de problema da base para treino."""
    g0, g1 = df[df[SENS]==0], df[df[SENS]==1]
    if len(g1) == 0 or len(g0) == 0:
        return 'sem_grupo'       # grupo inteiro ausente
    if g0[LABEL].nunique() < 2 or g1[LABEL].nunique() < 2:
        return 'monocl'          # algum grupo tem so 1 classe
    return 'ok'


def executar_fold(tk, bases):
    info = BASES[tk]
    bal = balancear(bases[tk])
    ds_tr, scaler = to_bld(bal, fit=True)

    modelos = []
    for eta in ETAS:
        pr = PrejudiceRemover(eta=eta, sensitive_attr=SENS, class_attr=LABEL)
        pr.fit(ds_tr)
        modelos.append(pr)

    res = []
    for test_k, test_info in BASES.items():
        if test_k == tk:
            continue
        tdf = bases[test_k]
        if len(tdf) > MAX_TESTE:
            tdf = tdf.sample(n=MAX_TESTE, random_state=SEED)
        ds_te, _ = to_bld(tdf, scaler=scaler)
        y_true = ds_te.labels.ravel()
        s_true = ds_te.protected_attributes.ravel()

        for eta, mod in zip(ETAS, modelos):
            try:
                preds = mod.predict(ds_te)
                y_pred = preds.labels.ravel()
            except (IndexError, ValueError):
                y_pred = np.ones_like(y_true)
            m = calc_metricas(y_true, y_pred, s_true)
            m.update(base_treino=tk, vies_treino=info[2], serie_treino=info[1],
                     base_teste=test_k, vies_teste=test_info[2], serie_teste=test_info[1],
                     eta=eta)
            res.append(m)
    return res


def medir_vies_dados(bases):
    """Calcula medidas de vies nos dados brutos (antes de qualquer modelo)."""
    registros = []
    for k, (arq, serie, vies) in BASES.items():
        df = bases[k]
        n = len(df)
        g_priv = df[df[SENS] == 1]   # grupo privilegiado
        g_desp = df[df[SENS] == 0]   # grupo desprivilegiado (original 4)
        n_priv, n_desp = len(g_priv), len(g_desp)

        # Class Imbalance (CI) = (n_priv - n_desp) / n_total
        ci = (n_priv - n_desp) / n if n > 0 else 0

        # Taxa de label favoravel (fraude=0) em cada grupo
        rate_priv = (g_priv[LABEL] == 0).mean() if n_priv > 0 else np.nan
        rate_desp = (g_desp[LABEL] == 0).mean() if n_desp > 0 else np.nan

        # Statistical Parity Difference nos dados (SPD_dados)
        spd_dados = (rate_desp - rate_priv) if (np.isfinite(rate_priv) and np.isfinite(rate_desp)) else np.nan

        # Disparate Impact (DI = rate_desp / rate_priv)
        di = (rate_desp / rate_priv) if (rate_priv and rate_priv > 0) else np.nan

        registros.append(dict(
            base=k, serie=serie, vies=vies,
            n_total=n, n_priv=n_priv, n_desp=n_desp,
            ci=ci, spd_dados=spd_dados, abs_spd_dados=abs(spd_dados) if np.isfinite(spd_dados) else np.nan,
            di=di
        ))
    return pd.DataFrame(registros)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    print('Carregando 10 bases...')
    bases = {}
    tipos = {}
    for k, (arq, serie, vies) in BASES.items():
        df = carregar(arq)
        bases[k] = df
        t = classificar_base(df)
        tipos[k] = t
        g0, g1 = df[df[SENS]==0], df[df[SENS]==1]
        tag = '' if t == 'ok' else f' [{t}]'
        print(f'  {k:12s} {len(df):>6d} reg  '
              f'g1={len(g1):>5d}(f{(g1[LABEL]==0).sum():>5d}/l{(g1[LABEL]==1).sum():>5d})  '
              f'g4={len(g0):>5d}(f{(g0[LABEL]==0).sum():>5d}/l{(g0[LABEL]==1).sum():>5d})'
              f'{tag}')

    bases_ok = [k for k, t in tipos.items() if t == 'ok']
    bases_deg = [k for k, t in tipos.items() if t != 'ok']
    todas = list(BASES.keys())

    # Medidas de vies nos dados brutos
    print('\nMedidas de vies nos dados brutos')
    df_vies = medir_vies_dados(bases)
    print(f'{"Base":>12s}  {"Serie":>4s}  {"Vies":>5s}  {"CI":>7s}  {"SPD_dados":>9s}  {"DI":>7s}')
    print('-' * 55)
    for _, row in df_vies.iterrows():
        di_str = f'{row.di:.3f}' if np.isfinite(row.di) else '  N/A'
        spd_str = f'{row.spd_dados:+.4f}' if np.isfinite(row.spd_dados) else '    N/A'
        print(f'{row.base:>12s}  {row.serie:>4s}  {row.vies:5.2f}  {row.ci:+7.4f}  {spd_str:>9s}  {di_str:>7s}')
    df_vies.to_csv(f'{OUT}/vies_dados.csv', index=False)
    print(f'  -> {OUT}/vies_dados.csv\n')

    print(f'10 ciclos de treino ({len(bases_deg)} degeneradas: {bases_deg})')
    print(f'Total: {len(todas)*9*len(ETAS)} experimentos\n')

    # Executar TODOS os 10 folds em paralelo
    print('Treinando...')
    resultados = Parallel(n_jobs=3, verbose=5)(
        delayed(executar_fold)(tk, bases) for tk in todas
    )

    todos = [r for fold in resultados for r in fold]
    df_res = pd.DataFrame(todos)
    df_res.to_csv(f'{OUT}/resultados.csv', index=False)

    # Resumo apenas dos ciclos validos
    df_ok = df_res[df_res['base_treino'].isin(bases_ok)]
    print(f'\n{"="*60}')
    print(f'Resumo (ciclos validos: {len(bases_ok)} bases)')
    print(f'{"eta":>5s}  {"F1":>7s}  {"+-":>6s}  {"|SPD|":>7s}  {"+-":>6s}  {"FR":>5s}')
    print(f'{"-"*60}')
    for eta, g in df_ok.groupby('eta'):
        print(f'{eta:5.0f}  {g.f1.mean():7.4f}  {g.f1.std():6.4f}  '
              f'{g.abs_spd.mean():7.4f}  {g.abs_spd.std():6.4f}  {g.fr.mean():5.3f}')

    # Resumo degenerados
    if bases_deg:
        df_deg = df_res[df_res['base_treino'].isin(bases_deg)]
        print(f'\nCiclos degenerados ({bases_deg}):')
        for eta, g in df_deg.groupby('eta'):
            print(f'  eta={eta:.0f}: F1={g.f1.mean():.4f} |SPD|={g.abs_spd.mean():.4f}')

    # Recomendacao
    resumo = df_ok.groupby('eta').agg(f1=('f1','mean'), spd=('abs_spd','mean'), fr=('fr','mean'))
    f1_max = resumo['f1'].max()
    cand = resumo[resumo['f1'] >= 0.8 * f1_max]
    eta_rec = cand['fr'].idxmax() if len(cand) > 0 else 0
    r = resumo.loc[eta_rec]
    print(f'\nRecomendacao: eta={eta_rec:.0f} (F1={r.f1:.4f}, |SPD|={r.spd:.4f}, FR={r.fr:.3f})')

    dt = time.time() - t0
    print(f'\nConcluido em {dt:.0f}s ({dt/60:.1f} min)')


if __name__ == '__main__':
    main()
