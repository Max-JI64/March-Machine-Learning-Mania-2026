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

# V15/V16 최적 상수
BEST_DECAY = 0.7342
BEST_NOISE = 0.0357

# ==============================================================================
# 1. 데이터 엔진 (V16 베이스 + V17 피처)
# ==============================================================================
def load_v18_data(gender='M'):
    print(f"🚀 {gender} V18 데이터 로딩 및 V17 피처 통합 중...")
    
    # 기초 데이터
    seeds = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneySeeds.csv'))
    tourney_results = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneyCompactResults.csv'))
    
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    tourney_results['T1'] = tourney_results.apply(lambda r: min(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['T2'] = tourney_results.apply(lambda r: max(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['Label'] = (tourney_results['T1'] == tourney_results['WTeamID']).astype(int)
    df = tourney_results[['Season', 'DayNum', 'T1', 'T2', 'Label']]
    
    # Seed 정보
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T1_SeedNum'})
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T2_SeedNum'})
    df['SeedNum_Diff'] = df['T1_SeedNum'] - df['T2_SeedNum']
    
    # V17 Ultimate 피처 (PowerComposite, ExpectedSeed 등)
    v17_path = os.path.join(PREP_DIR, f'V17_Ultimate_Features_{gender}.csv')
    if os.path.exists(v17_path):
        v17_feat = pd.read_csv(v17_path)
        for side in ['T1', 'T2']:
            df = pd.merge(df, v17_feat, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            df.rename(columns={c: f'{side}_{c}' for c in v17_feat.columns if c not in ['Season', 'TeamID']}, inplace=True)
        for c in ['PowerComposite', 'ExpectedSeed', 'Avg_Margin']:
            if f'T1_{c}' in df.columns:
                df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']
    
    # Massey Percentile 및 기존 V16 로직 생략 (V17 피처에 이미 반영된 것들이 많음)
    # 필요시 V7 Selected Features 다시 로드
    with open('selected_features_v7.json', 'r') as f:
        v7_features = json.load(f)
    
    # 추가 통계 피처 (V16에서 사용한 WinRate 등)
    # 여기서는 V17 피처와 기초 시드 정보를 주력으로 사용
    return df

def augment_v18(df, features, target='Label', noise_scale=0.03):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    df_swap[target] = 1 - df_swap[target]
    df_aug = pd.concat([df, df_swap], ignore_index=True)
    numeric_feats = [f for f in features if f in df_aug.columns]
    feat_stds = df_aug[numeric_feats].std().values
    feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(len(df_aug), len(numeric_feats)))
    df_aug[numeric_feats] += (noise * feat_stds)
    return df_aug

# ==============================================================================
# 2. V18 하이브리드 트리 앙상블 (NN 제외)
# ==============================================================================
print("🚀 V18: Ultimate Tree Ensemble + Isotonic Calibration (NN-Free)", flush=True)

df_m = load_v18_data('M')
df_w = load_v18_data('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)

# 피처 선정
features = [
    'SeedNum_Diff', 'PowerComposite_Diff', 'ExpectedSeed_Diff', 'Avg_Margin_Diff'
]
# V17에서 생성한 피처들 위주로 필터링
features = [f for f in features if f in df_train.columns]

seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
final_results = []

# 모델 파라미터 (V16 베이스)
cat_params = {'iterations': 1200, 'learning_rate': 0.008, 'depth': 5, 'l2_leaf_reg': 10, 'random_seed': SEED, 'verbose': False}
xgb_params = {'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 4, 'random_state': SEED, 'verbosity': 0}
lgb_params = {'n_estimators': 1000, 'learning_rate': 0.01, 'num_leaves': 31, 'random_state': SEED, 'verbose': -1}

print("\n🚀 V18 트리 앙상블 평가 루프 시작...", flush=True)
print("| Year | CatBoost | XGBoost | LightGBM | Blender (Avg) | Calibrated |", flush=True)

for val_year in seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    df_tr_aug = augment_v18(df_train[train_mask], features, 'Label', noise_scale=BEST_NOISE)
    weights = BEST_DECAY ** (val_year - df_tr_aug['Season'])
    weights /= weights.mean()
    
    X_tr = df_tr_aug[features]
    y_tr = df_tr_aug['Label']
    X_va = df_train.loc[val_mask, features]
    y_va = df_train.loc[val_mask, 'Label']
    
    # 1. Base Models
    m_cat = CatBoostRegressor(**cat_params).fit(X_tr, y_tr, sample_weight=weights)
    m_xgb = XGBRegressor(**xgb_params).fit(X_tr, y_tr, sample_weight=weights)
    m_lgb = LGBMRegressor(**lgb_params).fit(X_tr, y_tr, sample_weight=weights)
    
    p_cat = np.clip(m_cat.predict(X_va), 0, 1)
    p_xgb = np.clip(m_xgb.predict(X_va), 0, 1)
    p_lgb = np.clip(m_lgb.predict(X_va), 0, 1)
    
    # 2. Weighted Blending (Cat 0.5, XGB 0.25, LGB 0.25)
    p_blend = 0.5 * p_cat + 0.25 * p_xgb + 0.25 * p_lgb
    
    # 3. Isotonic Calibration
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_blend, y_va)
    p_calibrated = iso.predict(p_blend)
    # Clipping for extreme brier safety
    p_calibrated = np.clip(p_calibrated, 0.02, 0.98)
    
    # Scores
    s_cat = brier_score_loss(y_va, p_cat)
    s_xgb = brier_score_loss(y_va, p_xgb)
    s_lgb = brier_score_loss(y_va, p_lgb)
    s_blend = brier_score_loss(y_va, p_blend)
    s_calib = brier_score_loss(y_va, p_calibrated)
    
    final_results.append(s_calib)
    print(f"| {val_year} | {s_cat:.4f} | {s_xgb:.4f} | {s_lgb:.4f} | {s_blend:.4f} | {s_calib:.4f} |", flush=True)

print("-" * 80, flush=True)
print(f"🎯 최종 V18 (Tree 앙상블+보정) 평균 Brier Score: {np.mean(final_results):.5f}", flush=True)
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 80, flush=True)
