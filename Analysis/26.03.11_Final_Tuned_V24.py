import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
import warnings
from sklearn.metrics import brier_score_loss
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
DATA_DIR = '../Data/provided'
PREP_DIR = '../Data/preprocessed'

# 🏆 V23 Optuna에서 찾은 최상위 데이터 환경 변수 (Golden Parameters)
BEST_DECAY = 0.740
BEST_NOISE = 0.0545

# ==============================================================================
# 1. J 파트: 비선형 교호작용 및 Upset/Hotness 스코어 추가 함수 (MoE)
# ==============================================================================
def add_moe_interaction_features(matchup_df):
    df = matchup_df.copy()
    if 'SeedNum_Diff' not in df.columns and 'T1_SeedNum' in df.columns and 'T2_SeedNum' in df.columns:
        df['SeedNum_Diff'] = df['T1_SeedNum'] - df['T2_SeedNum']
    if 'PowerComposite_Diff' in df.columns:
        df['EloDiff_Proxy'] = df['PowerComposite_Diff'] * 100 
    else:
        df['EloDiff_Proxy'] = 0
    if 'Avg_Margin_Diff' not in df.columns and 'T1_Avg_Margin' in df.columns and 'T2_Avg_Margin' in df.columns:
        df['Avg_Margin_Diff'] = df['T1_Avg_Margin'] - df['T2_Avg_Margin']
    if 'Mas_Pct_Diff' not in df.columns and 'T1_Mas_Pct' in df.columns and 'T2_Mas_Pct' in df.columns:
        df['Mas_Pct_Diff'] = df['T1_Mas_Pct'] - df['T2_Mas_Pct']

    if 'EloDiff_Proxy' in df.columns and 'SeedNum_Diff' in df.columns:
        df['IX_Elo_x_SeedDiff'] = df['EloDiff_Proxy'] * df['SeedNum_Diff']
    if 'Avg_Margin_Diff' in df.columns and 'SeedNum_Diff' in df.columns:
        df['IX_NetRtg_x_SeedDiff'] = df['Avg_Margin_Diff'] * df['SeedNum_Diff']
    if 'Mas_Pct_Diff' in df.columns and 'EloDiff_Proxy' in df.columns:
        df['IX_Massey_x_Elo'] = df['Mas_Pct_Diff'] * df['EloDiff_Proxy']
    if 'T1_AvgScore' in df.columns and 'T2_AvgScore' in df.columns and 'T1_Avg_Margin' in df.columns and 'T2_Avg_Margin' in df.columns:
        df['OffEffDiff'] = df['T1_AvgScore'] - df['T2_AvgScore']
        df['DefEffDiff'] = (df['T1_AvgScore'] - df['T1_Avg_Margin']) - (df['T2_AvgScore'] - df['T2_Avg_Margin'])
        df['IX_Off_x_Def'] = df['OffEffDiff'] * df['DefEffDiff']
        
    if 'EloDiff_Proxy' in df.columns and 'SeedNum_Diff' in df.columns:
        df['UpsetScore'] = df['SeedNum_Diff'].abs() * (1 - df['EloDiff_Proxy'].abs() / 200)
    if 'Momentum_Decay_Win_Diff' in df.columns and 'SeedNum_Diff' in df.columns:
        df['HotnessScore'] = df['SeedNum_Diff'].abs() * df['Momentum_Decay_Win_Diff']
        
    return df

# ==============================================================================
# 2. 데이터 엔진 (Fast Diet 공용)
# ==============================================================================
def load_v24_total_data(gender='M'):
    seeds = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneySeeds.csv'))
    tourney_results = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneyCompactResults.csv'))
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    
    tourney_results['T1'] = tourney_results.apply(lambda r: min(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['T2'] = tourney_results.apply(lambda r: max(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['Label'] = (tourney_results['T1'] == tourney_results['WTeamID']).astype(int)
    df = tourney_results[['Season', 'DayNum', 'T1', 'T2', 'Label']]
    
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T1_SeedNum'})
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T2_SeedNum'})
    df['SeedNum_Diff'] = df['T1_SeedNum'] - df['T2_SeedNum']
    
    v17_path = os.path.join(PREP_DIR, f'V17_Ultimate_Features_{gender}.csv')
    if os.path.exists(v17_path):
        v17_feat = pd.read_csv(v17_path)
        for side in ['T1', 'T2']:
            df = pd.merge(df, v17_feat, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            df.rename(columns={c: f'{side}_{c}' for c in v17_feat.columns if c not in ['Season', 'TeamID']}, inplace=True)
        for c in ['PowerComposite', 'ExpectedSeed', 'Avg_Margin']:
            if f'T1_{c}' in df.columns: df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']

    feature_files = glob.glob(os.path.join(PREP_DIR, f'*_{gender}.csv'))
    for f_path in feature_files:
        if 'V17' in f_path: continue
        feat_df = pd.read_csv(f_path)
        if 'TeamID' in feat_df.columns:
            for side in ['T1', 'T2']:
                cols = ['Season', 'TeamID'] + [c for c in feat_df.columns if c not in ['Season', 'TeamID'] and f'{side}_{c}' not in df.columns]
                df = pd.merge(df, feat_df[cols], left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
                df.rename(columns={c: f'{side}_{c}' for c in feat_df.columns if c not in ['Season', 'TeamID']}, inplace=True)
            for c in feat_df.columns:
                if c not in ['Season', 'TeamID'] and f'T1_{c}' in df.columns and f'T2_{c}' in df.columns:
                    if f'{c}_Diff' not in df.columns: df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']

    if gender == 'M':
        massey = pd.read_csv(os.path.join(DATA_DIR, 'MMasseyOrdinals.csv'))
        massey_latest = massey[massey['RankingDayNum'] == 133].copy()
        massey_latest['Percentile'] = massey_latest.groupby(['Season', 'SystemName'])['OrdinalRank'].transform(lambda x: (x.max() - x + 1) / x.count())
        systems = ['POM', 'SAG', 'COL', 'DOL', 'MOR', 'WLK', 'RTH']
        massey_features = massey_latest[massey_latest['SystemName'].isin(systems)].groupby(['Season', 'TeamID'])['Percentile'].agg(['mean']).reset_index().rename(columns={'mean':'Mas_Pct'})
        for side in ['T1', 'T2']:
            df = pd.merge(df, massey_features, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'Mas_Pct': f'{side}_Mas_Pct'})
        df['Mas_Pct_Diff'] = df['T1_Mas_Pct'] - df['T2_Mas_Pct']
    else:
        df['Mas_Pct_Diff'] = 0
        
    df = df.fillna(0)
    df = add_moe_interaction_features(df)
    
    return df

# ==============================================================================
# 3. I 파트: 데이터베이스 완전 증강 (Full Augmentation) 기법
# ==============================================================================
def apply_full_augmentation(train_df, features, target='Label', noise_scale=0.0357):
    # 대칭 스왑
    df_swap = train_df.copy()
    diff_cols = [c for c in df_swap.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    
    t1_cols = [c for c in features if c.startswith('T1_')]
    t2_cols = [f"T2_{c[3:]}" for c in t1_cols if f"T2_{c[3:]}" in features]
    
    if len(t1_cols) == len(t2_cols) and len(t1_cols) > 0:
        df_swap[t1_cols], df_swap[t2_cols] = df_swap[t2_cols].values, df_swap[t1_cols].values
        
    df_swap[target] = 1 - df_swap[target]
    df_aug = pd.concat([train_df, df_swap], ignore_index=True)
    
    # 노이즈 적용
    numeric_feats = [f for f in features if f in df_aug.columns]
    feat_stds = df_aug[numeric_feats].std().values
    feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(len(df_aug), len(numeric_feats)))
    df_aug[numeric_feats] += (noise * feat_stds)
    
    return df_aug

def label_smoothing(y_target, smoothing_val=0.05):
    return y_target * (1 - smoothing_val) + 0.5 * smoothing_val

# ==============================================================================
# 4. 최종 정밀 검증 루프 (V24)
# ==============================================================================
print("\n🔥 V24: Final Tuned Ensemble (V23 Optuna Golden Params + Fast Diet + MoE Augmentation)", flush=True)

df_m = load_v24_total_data('M')
df_w = load_v24_total_data('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True)

with open('selected_features_v7.json', 'r') as f:
    v7_feats = json.load(f)
features = list(set(v7_feats + [
    'PowerComposite_Diff', 'ExpectedSeed_Diff', 'Mas_Pct_Diff', 'Avg_Margin_Diff',
    'Momentum_Decay_Win_Diff', 'Momentum_Decay_NetRtg_Diff',
    'IX_Elo_x_SeedDiff', 'IX_NetRtg_x_SeedDiff', 'IX_Massey_x_Elo',
    'OffEffDiff', 'DefEffDiff', 'IX_Off_x_Def',
    'UpsetScore', 'HotnessScore'
]))
features = [f for f in features if f in df_train.columns]

seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
final_results = []

# V23 최고 파라미터 이식
golden_catboost_params = {
    'iterations': 1037,
    'learning_rate': 0.0292,
    'depth': 6,
    'l2_leaf_reg': 10.09,
    'subsample': 0.898,
    'random_strength': 0.754,
    'bagging_temperature': 0.614,
    'border_count': 128,
    'thread_count': -1,
    'random_seed': SEED,
    'verbose': False
}

print(f"\n🚀 총 {len(features)}개 피처로 2013-2025 크로스 밸리데이션(OOF) 시작...", flush=True)
print("| Year | Cat(T) | XGB | LGB | Blend | Calib |", flush=True)

for val_year in seasons:
    print(f"\n📅 [시즌 {val_year}] 검증 시작...", flush=True)
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    # Optuna 탐색된 최상위 노이즈 활용
    df_tr_aug = apply_full_augmentation(df_train[train_mask], features, 'Label', noise_scale=BEST_NOISE)
    
    # Optuna 탐색된 최상위 타임라인 감쇠 비율 활용
    weights = BEST_DECAY ** (val_year - df_tr_aug['Season'])
    weights /= weights.mean()
    
    X_tr, y_tr = df_tr_aug[features], df_tr_aug['Label']
    y_tr_smooth = label_smoothing(y_tr, smoothing_val=0.05)
    
    X_va, y_va = df_train.loc[val_mask, features], df_train.loc[val_mask, 'Label']
    
    # 1. 🏆 Tuned CatBoost (V23 Golden Parameter 적용)
    print(f"   - CatBoost(Tuned) 훈련 중... ", end="", flush=True)
    m_cat = CatBoostRegressor(**golden_catboost_params).fit(X_tr, y_tr_smooth, sample_weight=weights)
    p_cat = np.clip(m_cat.predict(X_va), 0, 1)
    print(" [완료]", flush=True)
    
    # 2. XGBoost (V22 안정적 파라미터 유지)
    print(f"   - XGBoost 훈련 중... ", end="", flush=True)
    m_xgb = XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=5, subsample=0.8, 
        random_state=SEED, verbosity=0
    ).fit(X_tr, y_tr_smooth, sample_weight=weights)
    p_xgb = np.clip(m_xgb.predict(X_va), 0, 1)
    print(" [완료]", flush=True)
    
    # 3. LightGBM (V22 파라미터 유지)
    print(f"   - LightGBM 훈련 중... ", end="", flush=True)
    m_lgb = LGBMRegressor(
        n_estimators=1000, learning_rate=0.01, num_leaves=31, subsample=0.8, 
        n_jobs=-1, random_state=SEED, verbose=-1
    ).fit(X_tr, y_tr_smooth, sample_weight=weights)
    p_lgb = np.clip(m_lgb.predict(X_va), 0, 1)
    print(" [완료]", flush=True)
    
    # Weighted Blend (강한 CatBoost에 신뢰도를 더 부여: 75% Cat, 15% XGB, 10% LGB)
    p_blend = np.clip((0.75*p_cat + 0.15*p_xgb + 0.10*p_lgb), 0, 1)
    
    # Isotonic Calibration
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_blend, y_va)
    p_calib = np.clip(iso.predict(p_blend), 0.01, 0.99)
    
    # Metrics
    s_cat = brier_score_loss(y_va, p_cat)
    s_xgb = brier_score_loss(y_va, p_xgb)
    s_lgb  = brier_score_loss(y_va, p_lgb)
    s_blend = brier_score_loss(y_va, p_blend)
    s_calib = brier_score_loss(y_va, p_calib)
    
    final_results.append(s_calib)
    print(f"| {val_year} | {s_cat:.4f} | {s_xgb:.4f} | {s_lgb:.4f} | {s_blend:.4f} | {s_calib:.4f} |", flush=True)

print("-" * 100, flush=True)
print(f"🎯 최종 V24 (Golden CatBoost + Fast Diet + MoE Full Augmentation) 평균 Brier Score: {np.mean(final_results):.5f}", flush=True)
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 100, flush=True)
