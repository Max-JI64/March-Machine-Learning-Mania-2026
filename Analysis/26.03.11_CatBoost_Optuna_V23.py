import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
import warnings
import optuna
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor

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

# V15-V18 최적 상수 적용
BEST_DECAY = 0.7342
BEST_NOISE = 0.0357

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
# 2. 데이터 통합 (V22와 동일)
# ==============================================================================
def load_v23_total_data(gender='M'):
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
# 3. 데이터 증강 (I 파트)
# ==============================================================================
def apply_full_augmentation(train_df, features, target='Label', noise_scale=0.0357):
    df_swap = train_df.copy()
    diff_cols = [c for c in df_swap.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    
    t1_cols = [c for c in features if c.startswith('T1_')]
    t2_cols = [f"T2_{c[3:]}" for c in t1_cols if f"T2_{c[3:]}" in features]
    if len(t1_cols) == len(t2_cols) and len(t1_cols) > 0:
        df_swap[t1_cols], df_swap[t2_cols] = df_swap[t2_cols].values, df_swap[t1_cols].values
        
    df_swap[target] = 1 - df_swap[target]
    df_aug = pd.concat([train_df, df_swap], ignore_index=True)
    
    numeric_feats = [f for f in features if f in df_aug.columns]
    feat_stds = df_aug[numeric_feats].std().values
    feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(len(df_aug), len(numeric_feats)))
    df_aug[numeric_feats] += (noise * feat_stds)
    
    return df_aug

def label_smoothing(y_target, smoothing_val=0.05):
    return y_target * (1 - smoothing_val) + 0.5 * smoothing_val

# ==============================================================================
# 4. Optuna 딥튜닝 환경 세팅
# ==============================================================================
print("\n🔥 V23: CatBoost Deep Tuning w/ Advanced MoE & Full Augmentation (Silent Mode)", flush=True)

df_m = load_v23_total_data('M')
df_w = load_v23_total_data('W')
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

# 빠른 최적화를 위해 가장 최신 핵심 시즌(2025)만 단독 검증하여 튜닝 부하 대폭 감소
VAL_SEASONS = [2025]
print(f"🎯 Target Validation Seasons: {VAL_SEASONS} (Fast Tuning Mode)")

def objective(trial):
    # CatBoost 하이퍼파라미터 탐색 공간
    params = {
        'iterations': trial.suggest_int('iterations', 800, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 25),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_categorical('border_count', [128, 254]),
        'thread_count': -1,
        'random_seed': SEED,
        'early_stopping_rounds': 50, # 시간 단축의 핵심 (오버피팅 시 조기 종료)
        'verbose': False
    }
    
    noise_scale = trial.suggest_float('noise_scale', 0.01, 0.06)
    decay = trial.suggest_float('decay', 0.5, 0.8)
    
    cv_scores = []
    
    for val_year in VAL_SEASONS:
        train_mask = (df_train['Season'] < val_year)
        val_mask = (df_train['Season'] == val_year)
        
        df_tr_aug = apply_full_augmentation(df_train[train_mask], features, 'Label', noise_scale=noise_scale)
        weights = decay ** (val_year - df_tr_aug['Season'])
        weights /= weights.mean()
        
        X_tr, y_tr = df_tr_aug[features], df_tr_aug['Label']
        y_tr_smooth = label_smoothing(y_tr, smoothing_val=0.05)
        
        X_va, y_va = df_train.loc[val_mask, features], df_train.loc[val_mask, 'Label']
        
        # 모델 훈련 (eval_set 제공으로 early stopping 활성화)
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr_smooth, 
            sample_weight=weights,
            eval_set=(X_va, y_va),
            use_best_model=True
        )
        p_val = np.clip(model.predict(X_va), 0, 1)
        
        # Calibration (중요: 단일 모델 평가에서도 보정 사용)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p_val, y_va)
        p_calib = np.clip(iso.predict(p_val), 0.01, 0.99)
        
        score = brier_score_loss(y_va, p_calib)
        cv_scores.append(score)
        
    return np.mean(cv_scores)

print("\n🚀 Starting CatBoost Optuna Deep Tuning (30 Trials)...", flush=True)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("-" * 50)
print(f"🏆 Best Trial Brier Score: {study.best_value:.5f}")
print("🥇 Best Parameters:")
for key, value in study.best_params.items():
    print(f"    '{key}': {value},")
print("-" * 50)

# 결과 저장
with open('v23_catboost_best_params.json', 'w') as f:
    json.dump(study.best_params, f, indent=4)
print(f"⏱️ Total Tuning Time: {(time.time()-START_TIME)/60:.1f} min", flush=True)
