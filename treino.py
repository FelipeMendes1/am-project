"""
Pipeline de Treinamento e Validação.
Treina em cada base e avalia nas demais, registrando métricas de desempenho
(F1, Recall, Precision) e de justiça (SPD, DI) para cada nível de penalidade ETA.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample

# Configuração do caminho para o módulo Kamishima (AIF360)
import aif360
kam_path = os.path.join(os.path.dirname(aif360.__file__), 'algorithms', 'inprocessing', 'kamfadm-2012ecmlpkdd')
if kam_path not in sys.path:
    sys.path.insert(0, kam_path)
from fadm.lr.pr import LRwPRType4

# Configurações globais
ETAS = [0.0, 1.0, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0]

N_TREINO = 10000
MAX_TESTE = 20000
SENSITIVE = 'RAMO_ATIVIDADE_1'
TARGET = 'I-d'
SEED = 42

PRIV_LABEL  = 1   # Ramo 1 → grupo privilegiado
DEPRIV_ORIG = 4   # Ramo 4 → remapeado para 0

OUT_DIR  = './resultados'
DATA_DIR = './Equidade - bases de dados públicas'

BASES_MAP = {
    'pd_v0': ('pd_v0.csv', 'PD', 0.0), 'pd_v0.25': ('pd_v0_25.csv', 'PD', 0.25),
    'pd_v0.5': ('pd_v0_5.csv', 'PD', 0.5), 'pd_v0.75': ('pd_v0_75.csv', 'PD', 0.75),
    'pd_v1': ('pd_v1.csv', 'PD', 1.0), 'gs_v0': ('gs_v0.csv', 'GS', 0.0),
    'gs_v0.25': ('gs_v0_25.csv', 'GS', 0.25), 'gs_v0.5': ('gs_v0_5.csv', 'GS', 0.5),
    'gs_v0.75': ('gs_v0_75.csv', 'GS', 0.75), 'gs_v1': ('gs_v1.csv', 'GS', 1.0)
}

FEATURES = [
    'final_de_semana',             # 1 se sáb ou dom, 0 caso contrário
    'pct_saldo_gasto',             # VALOR_TRANSACAO / (|VALOR_SALDO| + 1)
    'zscore_valor_no_titular',     # z-score do valor em relação ao histórico expanding do titular
    'n_transacoes_dia_tit',        # contagem acumulada de transações do titular no dia
    'is_cnab220',                  # 1 se CNAB=220 (equivalente a is_credito no dataset)
    'prop_cnab220_tit',            # proporção histórica de CNAB220 do titular (expanding, shift=1)
    'zscore_valor_no_ramo',        # z-score do valor dentro do ramo de atividade
    'zscore_n_trans_mes_no_ramo',  # z-score do volume mensal de transações dentro do ramo
    'zscore_reg_no_ramo',          # z-score da regularidade do titular dentro do ramo
    'zscore_valor_total_mes_no_ramo', # z-score do volume financeiro mensal dentro do ramo
]

# Apenas as features contínuas recebem StandardScaler (binárias ficam como estão)
CONT_FEATURES = [
    'pct_saldo_gasto', 'zscore_valor_no_titular', 'n_transacoes_dia_tit',
    'prop_cnab220_tit', 'zscore_valor_no_ramo', 'zscore_n_trans_mes_no_ramo',
    'zscore_reg_no_ramo', 'zscore_valor_total_mes_no_ramo',
]

class PrejudiceRemoverModel(BaseEstimator, ClassifierMixin):
    # Fallback para Regressão Logística quando ETA=0 ou quando os dados
    # não têm os dois grupos e as duas classes necessárias para o PR.
    def __init__(self, eta=1.0):
        self.eta = eta
        self.model = None
        self.fallback = None
        self.use_pr = False

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.is_fitted_ = True

        X_feat = X[:, :-1]
        s_col  = X[:, -1]

        if self.eta == 0.0:
            self.fallback = LogisticRegression(max_iter=1000, random_state=SEED)
            self.fallback.fit(X_feat, y)
            return self

        if len(np.unique(s_col)) == 2 and len(np.unique(y)) == 2:
            self.use_pr = True
            self.model = LRwPRType4(eta=self.eta, C=1.0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model.fit(X.astype(np.float32), y, ns=1, itype=3)
            except Exception:
                self.use_pr = False
                self.fallback = LogisticRegression(max_iter=1000, random_state=SEED)
                self.fallback.fit(X_feat, y)
        else:
            self.fallback = LogisticRegression(max_iter=1000, random_state=SEED)
            self.fallback.fit(X_feat, y)

        return self

    def predict(self, X):
        if self.use_pr:
            return np.argmax(self.model.predict_proba(X.astype(np.float32)), axis=1)
        return self.fallback.predict(X[:, :-1])


def carregar_e_preparar_dados(nome_base):
    path = os.path.join(DATA_DIR, BASES_MAP[nome_base][0])
    df = pd.read_csv(path, low_memory=False)

    df = df[df[SENSITIVE].isin([PRIV_LABEL, DEPRIV_ORIG])].copy()
    df[SENSITIVE] = df[SENSITIVE].replace({DEPRIV_ORIG: 0}).astype(np.int8)

    # Ordenação cronológica necessária para as expanding windows sem leakage
    df['DATA_LANCAMENTO'] = pd.to_datetime(df['DATA_LANCAMENTO'], errors='coerce')
    df = df.dropna(subset=['DATA_LANCAMENTO']).sort_values(['CPF_CNPJ_TITULAR', 'DATA_LANCAMENTO'])

    df['dia_da_semana']   = df['DATA_LANCAMENTO'].dt.dayofweek.astype(np.int8)
    df['final_de_semana'] = df['dia_da_semana'].isin([5, 6]).astype(np.int8)
    df['pct_saldo_gasto'] = (df['VALOR_TRANSACAO'] / (df['VALOR_SALDO'].abs() + 1)).astype(np.float32)

    df['data_dia'] = df['DATA_LANCAMENTO'].dt.date
    df['mes']      = df['DATA_LANCAMENTO'].dt.tz_localize(None).dt.to_period('M')

    grp_tit_mes = df.groupby(['CPF_CNPJ_TITULAR', 'mes'])
    df['n_transacoes_mes_tit'] = grp_tit_mes.cumcount() + 1
    df['valor_total_mes_tit']  = grp_tit_mes['VALOR_TRANSACAO'].cumsum().astype(np.float32)

    grp_tit_dia = df.groupby(['CPF_CNPJ_TITULAR', 'data_dia'])
    df['n_transacoes_dia_tit'] = grp_tit_dia.cumcount() + 1

    df['novo_dia']         = (~df.duplicated(subset=['CPF_CNPJ_TITULAR', 'mes', 'data_dia'])).astype(int)
    df['dias_ativos_mes']  = df.groupby(['CPF_CNPJ_TITULAR', 'mes'])['novo_dia'].cumsum()
    df['regularidade_tit'] = (df['dias_ativos_mes'] / df['DATA_LANCAMENTO'].dt.day).astype(np.float32)

    grp_tit   = df.groupby('CPF_CNPJ_TITULAR')['VALOR_TRANSACAO']
    hist_mean = grp_tit.expanding().mean().reset_index(level=0, drop=True)
    hist_std  = grp_tit.expanding().std().reset_index(level=0, drop=True).fillna(0)
    df['zscore_valor_no_titular'] = np.where(
        hist_std > 0, (df['VALOR_TRANSACAO'] - hist_mean) / hist_std, 0.0
    ).clip(-10, 10).astype(np.float32)

    df['is_cnab220'] = (df['CNAB'] == 220).astype(np.int8)
    grp_tit_cnab = df.groupby('CPF_CNPJ_TITULAR')['is_cnab220']
    df['prop_cnab220_tit'] = grp_tit_cnab.apply(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    ).values.astype(np.float32)

    # Z-scores por ramo: removem o efeito de grupo mantendo o sinal de anomalia
    def zscore_por_ramo(col, clip=5):
        mu  = df.groupby(SENSITIVE)[col].transform('mean')
        std = df.groupby(SENSITIVE)[col].transform('std').fillna(0)
        return ((df[col] - mu) / (std + 1e-6)).clip(-clip, clip).astype(np.float32)

    df['zscore_valor_no_ramo']           = zscore_por_ramo('VALOR_TRANSACAO', clip=10)
    df['zscore_n_trans_mes_no_ramo']     = zscore_por_ramo('n_transacoes_mes_tit')
    df['zscore_reg_no_ramo']             = zscore_por_ramo('regularidade_tit')
    df['zscore_valor_total_mes_no_ramo'] = zscore_por_ramo('valor_total_mes_tit')

    return df[FEATURES + [SENSITIVE, TARGET]].fillna(0)


def calcular_vies_dados_iniciais():
    registros = []
    for nome, (arq, serie, vies_v) in BASES_MAP.items():
        df = carregar_e_preparar_dados(nome)
        mask_priv   = df[SENSITIVE] == 1
        mask_depriv = df[SENSITIVE] == 0

        # Classe favorável é TARGET=0 (transação normal)
        taxa_p = (df.loc[mask_priv,   TARGET] == 0).mean() if mask_priv.any()   else np.nan
        taxa_u = (df.loc[mask_depriv, TARGET] == 0).mean() if mask_depriv.any() else np.nan

        spd = (taxa_u - taxa_p) if not (np.isnan(taxa_u) or np.isnan(taxa_p)) else np.nan
        di  = (taxa_u / taxa_p) if (taxa_p and taxa_p > 0 and not np.isnan(taxa_u)) else np.nan

        registros.append({
            'base': nome, 'serie': serie, 'vies': vies_v,
            'spd_dados': spd, 'abs_spd_dados': abs(spd) if not np.isnan(spd) else np.nan,
            'di_dados': di
        })
    return pd.DataFrame(registros)


def treinar_e_avaliar_base(nome_treino):
    df_train = carregar_e_preparar_dados(nome_treino)

    n_amostras = min(len(df_train[df_train[TARGET]==0]),
                     len(df_train[df_train[TARGET]==1]),
                     N_TREINO // 2)
    df_bal = pd.concat([
        resample(df_train[df_train[TARGET]==0], n_samples=n_amostras, replace=False, random_state=SEED),
        resample(df_train[df_train[TARGET]==1], n_samples=n_amostras, replace=False, random_state=SEED)
    ]).sample(frac=1, random_state=SEED)

    X_train_raw = df_bal[FEATURES + [SENSITIVE]].values
    y_train     = df_bal[TARGET].values

    # Scaler seletivo: apenas features contínuas (binárias ficam como estão)
    cont_indices = [FEATURES.index(c) for c in CONT_FEATURES if c in FEATURES]
    scaler  = StandardScaler()
    X_train = X_train_raw.copy().astype(np.float64)
    if cont_indices:
        X_train[:, cont_indices] = scaler.fit_transform(X_train_raw[:, cont_indices])

    modelos = {}
    for eta in ETAS:
        modelos[eta] = PrejudiceRemoverModel(eta=eta)
        modelos[eta].fit(X_train, y_train)

    resultados = []

    for test_k in BASES_MAP:
        if test_k == nome_treino:
            continue

        df_te = carregar_e_preparar_dados(test_k)
        if len(df_te) > MAX_TESTE:
            df_te = df_te.sample(n=MAX_TESTE, random_state=SEED)

        X_test_raw = df_te[FEATURES + [SENSITIVE]].values
        y_test     = df_te[TARGET].values
        s_test     = df_te[SENSITIVE].values

        X_test = X_test_raw.copy().astype(np.float64)
        if cont_indices:
            X_test[:, cont_indices] = scaler.transform(X_test_raw[:, cont_indices])

        for eta in ETAS:
            y_pred  = modelos[eta].predict(X_test)
            mask_u  = (s_test == 0)
            mask_p  = (s_test == 1)

            # SPD e DI calculados sobre a classe favorável (TARGET=0, transação normal)
            if mask_u.any() and mask_p.any():
                taxa_u = (y_pred[mask_u] == 0).mean()
                taxa_p = (y_pred[mask_p] == 0).mean()
                spd = taxa_u - taxa_p
                di  = (taxa_u / taxa_p) if taxa_p > 0 else np.nan
            else:
                spd, di = np.nan, np.nan

            resultados.append({
                'base_treino': nome_treino, 'serie_treino': BASES_MAP[nome_treino][1],
                'vies_treino': BASES_MAP[nome_treino][2], 'base_teste': test_k,
                'serie_teste': BASES_MAP[test_k][1], 'vies_teste': BASES_MAP[test_k][2],
                'eta': eta, 'usou_modelo_padrao': not modelos[eta].use_pr,
                'f1':        f1_score(y_test, y_pred, zero_division=0),
                'recall':    recall_score(y_test, y_pred, zero_division=0),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'spd': spd, 'abs_spd': abs(spd) if not np.isnan(spd) else np.nan, 'di': di,
                'n_priv_teste': int(mask_p.sum()), 'n_depriv_teste': int(mask_u.sum()),
            })

    return resultados

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    print("Calculando viés original das bases de dados...")
    df_vies = calcular_vies_dados_iniciais()
    df_vies.to_csv(os.path.join(OUT_DIR, 'vies_original.csv'), index=False)

    print("Iniciando treinamento e validação...")
    jobs = Parallel(n_jobs=-1)(delayed(treinar_e_avaliar_base)(k) for k in BASES_MAP)

    df_res = pd.DataFrame([item for sub in jobs for item in sub])
    df_res.to_csv(os.path.join(OUT_DIR, 'resultados.csv'), index=False)

    print(f"\nTreinamento concluído em {time.time() - t_start:.1f} segundos.")
    print("\nMédias por ETA:")
    print(df_res.groupby('eta')[['f1', 'recall', 'abs_spd', 'di']].mean().round(4).to_string())

if __name__ == '__main__':
    main()
