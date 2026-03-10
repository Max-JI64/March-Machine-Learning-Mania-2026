import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
import warnings
from sklearn.metrics import brier_score_loss
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
N_JOBS = min(os.cpu_count(), 64)

DATA_DIR = '../Data/provided'
PREP_DIR = '../Data/preprocessed'

# ==============================================================================
# 1. 1985년부터의 전체 데이터 통합 로드 함수
# ==============================================================================
def load_all_time_data(gender='M'):
    print(f"🚀 {gender} 1985년~현재 전 기간 데이터 로드 중...")
    
    # 기초 데이터 로드 (시드 및 토너먼트 결과)
    seeds = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneySeeds.csv'))
    tourney_results = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneyCompactResults.csv'))
    
    # 시드 숫자 추출
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    
    # 매치업 생성 (T1 < T2 규칙)
    tourney_results['T1'] = tourney_results.apply(lambda r: min(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['T2'] = tourney_results.apply(lambda r: max(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['Label'] = (tourney_results['T1'] == tourney_results['WTeamID']).astype(int)
    
    # 2013년 이후 검증을 위해 1985년부터의 모든 토너먼트 매치업 사용
    df = tourney_results[['Season', 'DayNum', 'T1', 'T2', 'Label']]
    
    # 시드 결합
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T1_SeedNum'})
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T2_SeedNum'})
    df['SeedDiff'] = df['T1_SeedNum'] - df['T2_SeedNum']
    
    # 정규 시즌 통계 (1985년부터 집계)
    reg_df = pd.read_csv(os.path.join(DATA_DIR, f'{gender}RegularSeasonCompactResults.csv'))
    
    # 시즌별 승률/평균득점/평균마진 계산
    w_stats = reg_df.groupby(['Season', 'WTeamID']).agg(WScore=('WScore', 'sum'), LScore=('LScore', 'sum'), Wins=('WTeamID', 'count'), Games=('WTeamID', 'count')).reset_index().rename(columns={'WTeamID':'TeamID'})
    l_stats = reg_df.groupby(['Season', 'LTeamID']).agg(LScore=('LScore', 'sum'), WScore=('WScore', 'sum'), Wins=('LTeamID', 'size'), Games=('LTeamID', 'count')).reset_index().rename(columns={'LTeamID':'TeamID'})
    l_stats['Wins'] = 0 # 패배 데이터셋이므로 승수는 0
    
    season_stats = pd.concat([w_stats, l_stats]).groupby(['Season', 'TeamID']).sum().reset_index()
    season_stats['WinRate'] = season_stats['Wins'] / season_stats['Games']
    season_stats['AvgScore'] = season_stats['WScore'] / season_stats['Games']
    season_stats['AvgMargin'] = (season_stats['WScore'] - season_stats['LScore']) / season_stats['Games']
    
    # 매치업에 통계 결합
    for side in ['T1', 'T2']:
        df = pd.merge(df, season_stats[['Season', 'TeamID', 'WinRate', 'AvgScore', 'AvgMargin']], left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
        df.rename(columns={c: f'{side}_{c}' for c in ['WinRate', 'AvgScore', 'AvgMargin']}, inplace=True)
    
    df['WinRateDiff'] = df['T1_WinRate'] - df['T2_WinRate']
    df['AvgMarginDiff'] = df['T1_AvgMargin'] - df['T2_AvgMargin']
    
    # Massey Ordinals (2003년부터만 존재하므로 1985-2002는 보정)
    if gender == 'M':
        massey = pd.read_csv(os.path.join(DATA_DIR, 'MMasseyOrdinals.csv'))
        # 시즌 마지막 랭킹 (DayNum 133 기준)
        last_massey = massey[massey['RankingDayNum'] == 133].groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()
        df['Is_Pre_Massey'] = (df['Season'] < 2003).astype(int)
        
        for side in ['T1', 'T2']:
            df = pd.merge(df, last_massey, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            # 2003년 이전은 전체 평균(약 160위)으로 채움
            df['OrdinalRank'] = df['OrdinalRank'].fillna(160)
            df.rename(columns={'OrdinalRank': f'{side}_MOR'}, inplace=True)
        df['MORDiff'] = df['T1_MOR'] - df['T2_MOR']
    else:
        df['Is_Pre_Massey'] = 0 # 여성부는 Massey 없음
        df['MORDiff'] = 0
        
    return df

# ==============================================================================
# 2. V11 모멘텀 피처 병합 함수
# ==============================================================================
def merge_momentum(df, gender='M'):
    mom_file = os.path.join(PREP_DIR, f'momentum_form_features_{gender}.csv')
    if os.path.exists(mom_file):
        mom_df = pd.read_csv(mom_file)
        for side in ['T1', 'T2']:
            df = pd.merge(df, mom_df, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            df.rename(columns={c: f'{side}_{c}' for c in mom_df.columns if c not in ['Season', 'TeamID']}, inplace=True)
        for c in mom_df.columns:
            if c not in ['Season', 'TeamID']:
                df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']
    return df

def augment_v13(df, features, target='Label', noise_scale=0.03):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    df_swap[target] = 1 - df_swap[target]
    df_augmented = pd.concat([df, df_swap], ignore_index=True)
    numeric_feats = [f for f in features if f in df_augmented.columns]
    feat_stds = df_augmented[numeric_feats].std().values
    noise = np.random.normal(0, noise_scale, size=(len(df_augmented), len(numeric_feats)))
    df_augmented[numeric_feats] += (noise * feat_stds)
    return df_augmented

# ==============================================================================
# 3. 메인 실행부
# ==============================================================================
print("🚀 V13: CatBoost Optimization with 1985-2025 Data", flush=True)

df_m = load_all_time_data('M')
df_m = merge_momentum(df_m, 'M')

df_w = load_all_time_data('W')
df_w = merge_momentum(df_w, 'W')

# 공통 피처 추출 및 통합
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)

# Interaction 피처 (V8 기반)
df_train['Seed_x_WinRate_Diff'] = df_train['SeedDiff'] * df_train['WinRateDiff']
df_train['Seed_x_Margin_Diff'] = df_train['SeedDiff'] * df_train['AvgMarginDiff']

# 피처 목록
drop_cols = ['Season', 'T1', 'T2', 'Label'] + [c for c in df_train.columns if c.startswith('T1_') or c.startswith('T2_')]
all_features = sorted([c for c in df_train.columns if c not in drop_cols])

print(f"📊 총 학습 가능 피처 수: {len(all_features)}", flush=True)
print(f"📊 전 기간 총 데이터 행 수: {len(df_train)} (1985년~2025년)", flush=True)

# 4. 자동 변수 선택 (1985년부터의 데이터를 기반으로 중요도 측정)
print("📊 자동 변수 선택 진행 중...", flush=True)
# 최신 경향 반영을 위해 2010년 이후 데이터로 중요도 측정
imp_mask = (df_train['Season'] >= 2010) & (df_train['Season'] != 2020)
model_imp = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=5, verbose=False, thread_count=N_JOBS, random_seed=SEED)
model_imp.fit(df_train.loc[imp_mask, all_features], df_train.loc[imp_mask, 'Label'])
selected_features = pd.Series(model_imp.get_feature_importance(), index=all_features).sort_values(ascending=False).head(40).index.tolist()

print(f"✅ 선택된 상위 10개 피처: {selected_features[:10]}", flush=True)

# 5. 최종 2013-2025 풀 시즌 검증 (V13)
cat_params = {
    'iterations': 2000, 'learning_rate': 0.015, 'depth': 6,
    'l2_leaf_reg': 7.0, 'subsample': 0.8, 'colsample_bylevel': 0.8,
    'random_seed': SEED, 'loss_function': 'RMSE', 'bootstrap_type': 'Bernoulli',
    'verbose': False, 'thread_count': N_JOBS
}

full_val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
final_scores = []

print("\n🚀 V13 풀 시즌 평가 시작 (1985년 데이터 포함 학습)...", flush=True)
for val_year in full_val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    # 80년대 데이터를 포함한 학습 데이터 구성
    X_tr = df_train.loc[train_mask, selected_features + ['Label', 'Season']]
    X_tr_aug = augment_v13(X_tr, selected_features)
    
    # 연도별 가중치 (최신 데이터에 더 큰 비중)
    weights = 0.70 ** (val_year - X_tr_aug['Season'])
    weights = weights / weights.mean()
    
    model = CatBoostRegressor(**cat_params)
    model.fit(X_tr_aug[selected_features], X_tr_aug['Label'], sample_weight=weights)
    
    preds = np.clip(model.predict(df_train.loc[val_mask, selected_features]), 0, 1)
    score = brier_score_loss(df_train.loc[val_mask, 'Label'], preds)
    final_scores.append(score)
    print(f"| {val_year} | {score:.4f} |", flush=True)

print("-" * 60, flush=True)
print(f"🎯 최종 V13 (1985년 데이터 통합) 평균 Brier Score: {np.mean(final_scores):.4f}", flush=True)
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 60, flush=True)
