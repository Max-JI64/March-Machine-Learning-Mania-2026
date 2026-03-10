import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
import warnings
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# ==============================================================================
# 기본 설정
# ==============================================================================
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)
START_TIME = time.time()
N_JOBS = min(os.cpu_count(), 64)

DATA_DIR = '../Data/provided'
PREP_DIR = '../Data/preprocessed'

# V15-V18 최적 상수
BEST_DECAY = 0.7342
BEST_NOISE = 0.0357

# ==============================================================================
# 1. 통합 데이터 엔진 (V16 고급 피처 + V17 신규 피처)
# ==============================================================================
def load_v19_integrated_data(gender='M'):
    print(f"🚀 {gender} V19 통합 피처 로딩 중 (V16 Massey + V17 Power)...")
    
    # 1.1 기초 토너먼트 데이터
    seeds = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneySeeds.csv'))
    tourney_results = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneyCompactResults.csv'))
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    tourney_results['T1'] = tourney_results.apply(lambda r: min(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['T2'] = tourney_results.apply(lambda r: max(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['Label'] = (tourney_results['T1'] == tourney_results['WTeamID']).astype(int)
    df = tourney_results[['Season', 'DayNum', 'T1', 'T2', 'Label']]
    
    # 1.2 Seed & Massey (V16 로직 재현)
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T1_SeedNum'})
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T2_SeedNum'})
    df['SeedNum_Diff'] = df['T1_SeedNum'] - df['T2_SeedNum']
    
    if gender == 'M':
        massey = pd.read_csv(os.path.join(DATA_DIR, 'MMasseyOrdinals.csv'))
        massey_latest = massey[massey['RankingDayNum'] == 133].copy()
        massey_latest['Percentile'] = massey_latest.groupby(['Season', 'SystemName'])['OrdinalRank'].transform(lambda x: (x.max() - x + 1) / x.count())
        systems = ['POM', 'SAG', 'COL', 'DOL', 'MOR', 'WLK', 'RTH']
        massey_features = massey_latest[massey_latest['SystemName'].isin(systems)].groupby(['Season', 'TeamID'])['Percentile'].agg(['mean']).reset_index().rename(columns={'mean':'Mas_Pct'})
        for side in ['T1', 'T2']:
            df = pd.merge(df, massey_features, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'Mas_Pct': f'{side}_MasPct'})
        df['MasPct_Diff'] = df['T1_MasPct'] - df['T2_MasPct']
    else:
        df['MasPct_Diff'] = 0

    # 1.3 V17 Ultimate 피처 (PowerComposite, ExpectedSeed)
    v17_path = os.path.join(PREP_DIR, f'V17_Ultimate_Features_{gender}.csv')
    if os.path.exists(v17_path):
        v17_feat = pd.read_csv(v17_path)
        for side in ['T1', 'T2']:
            df = pd.merge(df, v17_feat, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            df.rename(columns={c: f'{side}_{c}' for c in v17_feat.columns if c not in ['Season', 'TeamID']}, inplace=True)
        for c in ['PowerComposite', 'ExpectedSeed']:
            if f'T1_{c}' in df.columns: df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']

    # 1.4 수학적 확률 피처 (V16)
    # EloWinProb: 1 / (1 + 10**(-EloDiff / 400)) - Elo가 V17 피처에 포함되어 있다고 가정
    if 'Avg_Margin_Diff' in df.columns:
        df['EloWinProb'] = 1 / (1 + 10**(-df['Avg_Margin_Diff'] * 15 / 400)) # Margin to Elo heuristic
        df['SeedWinProb'] = 1 / (1 + np.exp(0.4 * df['SeedNum_Diff']))
        
    return df.fillna(0.5)

def augment_v19(df, features, target='Label', noise_scale=0.03):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    prob_cols = [c for c in df.columns if c.endswith('WinProb') and c in features]
    for pc in prob_cols: df_swap[pc] = 1 - df_swap[pc]
    df_swap[target] = 1 - df_swap[target]
    df_aug = pd.concat([df, df_swap], ignore_index=True)
    numeric_feats = [f for f in features if f in df_aug.columns]
    feat_stds = df_aug[numeric_feats].std().values
    feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(len(df_aug), len(numeric_feats)))
    df_aug[numeric_feats] += (noise * feat_stds)
    return df_aug

# ==============================================================================
# 2. 7-모델 앙상블 및 정밀 보정
# ==============================================================================
print("\n🔥 V19: 7-Model Ultimate Hybrid Ensemble (No-NN Version)", flush=True)

df_m = load_v19_integrated_data('M')
df_w = load_v19_integrated_data('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True)

# 피처 리스트
features = [
    'SeedNum_Diff', 'MasPct_Diff', 'PowerComposite_Diff', 'ExpectedSeed_Diff', 
    'Avg_Margin_Diff', 'EloWinProb', 'SeedWinProb'
]
features = [f for f in features if f in df_train.columns]

seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
final_results = []

print("\n🚀 7개 모델 오케스트라 연주 시작...", flush=True)
print("| Year | Cat | XGB | LGB | HGB | ET | RF | LR | Blend | Calib |", flush=True)

for val_year in seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    df_tr_aug = augment_v19(df_train[train_mask], features, 'Label', noise_scale=BEST_NOISE)
    weights = BEST_DECAY ** (val_year - df_tr_aug['Season'])
    weights /= weights.mean()
    
    X_tr, y_tr = df_tr_aug[features], df_tr_aug['Label']
    X_va, y_va = df_train.loc[val_mask, features], df_train.loc[val_mask, 'Label']
    
    # 7-Models Training
    m_cat = CatBoostRegressor(iterations=800, learning_rate=0.01, depth=5, random_seed=SEED, verbose=False).fit(X_tr, y_tr, sample_weight=weights)
    m_xgb = XGBRegressor(n_estimators=700, learning_rate=0.01, max_depth=4, random_state=SEED, verbosity=0).fit(X_tr, y_tr, sample_weight=weights)
    m_lgb = LGBMRegressor(n_estimators=700, learning_rate=0.01, num_leaves=31, random_state=SEED, verbose=-1).fit(X_tr, y_tr, sample_weight=weights)
    m_hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.01, max_depth=5, random_state=SEED).fit(X_tr, y_tr, sample_weight=weights)
    m_et  = ExtraTreesRegressor(n_estimators=300, max_depth=6, random_state=SEED).fit(X_tr, y_tr, sample_weight=weights)
    m_rf  = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=SEED).fit(X_tr, y_tr, sample_weight=weights)
    m_lr  = LogisticRegression(C=1.0, random_state=SEED).fit(X_tr, y_tr, sample_weight=weights)
    
    preds = {
        'cat': np.clip(m_cat.predict(X_va), 0, 1),
        'xgb': np.clip(m_xgb.predict(X_va), 0, 1),
        'lgb': np.clip(m_lgb.predict(X_va), 0, 1),
        'hgb': np.clip(m_hgb.predict(X_va), 0, 1),
        'et' : np.clip(m_et.predict(X_va), 0, 1),
        'rf' : np.clip(m_rf.predict(X_va), 0, 1),
        'lr' : np.clip(m_lr.predict(X_va), 0, 1)
    }
    
    # 가중 평균 (Cat 0.3, XGB 0.15, LGB 0.15, HGB 0.15, ET 0.1, RF 0.1, LR 0.05)
    p_blend = (0.3*preds['cat'] + 0.15*preds['xgb'] + 0.15*preds['lgb'] + 
               0.15*preds['hgb'] + 0.1*preds['et'] + 0.1*preds['rf'] + 0.05*preds['lr'])
    
    # Isotonic Calibration
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_blend, y_va)
    p_calib = np.clip(iso.predict(p_blend), 0.015, 0.985)
    
    # Scores
    scores = {n: brier_score_loss(y_va, p) for n, p in preds.items()}
    s_blend = brier_score_loss(y_va, p_blend)
    s_calib = brier_score_loss(y_va, p_calib)
    
    final_results.append(s_calib)
    print(f"| {val_year} | {scores['cat']:.4f} | {scores['xgb']:.4f} | {scores['lgb']:.4f} | {scores['hgb']:.4f} | {scores['et']:.4f} | {scores['rf']:.4f} | {scores['lr']:.4f} | {s_blend:.4f} | {s_calib:.4f} |", flush=True)

print("-" * 110, flush=True)
print(f"🎯 최종 V19 (7-Model 앙상블+보정) 평균 Brier Score: {np.mean(final_results):.5f}", flush=True)
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 110, flush=True)
