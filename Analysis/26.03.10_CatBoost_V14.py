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
# 1. 파트 I: Advanced Ratings 함수 (V8 복구)
# ==============================================================================
def compute_advanced_elo(df_games_sorted, K=20, HOME_ADV=75, REVERSION=0.75):
    elo_dict = {}
    elo_records = []
    current_season = df_games_sorted['Season'].iloc[0] if len(df_games_sorted) > 0 else 2000
    for idx, row in df_games_sorted.iterrows():
        season = row['Season']
        w_team = row['WTeamID']
        l_team = row['LTeamID']
        w_loc = row.get('WLoc', 'N')
        margin = row['WScore'] - row['LScore']
        if season != current_season:
            for t in elo_dict.keys(): elo_dict[t] = 1500 * (1 - REVERSION) + elo_dict[t] * REVERSION
            current_season = season
        elo_w = elo_dict.get(w_team, 1500)
        elo_l = elo_dict.get(l_team, 1500)
        elo_w_adj = elo_w + HOME_ADV if w_loc == 'H' else (elo_w - HOME_ADV if w_loc == 'A' else elo_w)
        expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w_adj) / 400.0))
        margin_multiplier = np.log(min(abs(margin), 25) + 1) / np.log(26)
        update = K * margin_multiplier * (1.0 - expected_w)
        elo_dict[w_team] = elo_w + update
        elo_dict[l_team] = elo_l - update
        elo_records.append({'Season': season, 'DayNum': row['DayNum'], 'WTeamID': w_team, 'LTeamID': l_team, 'W_Elo_Post': elo_dict[w_team], 'L_Elo_Post': elo_dict[l_team]})
    return pd.DataFrame(elo_records), elo_dict

def compute_srs(df_season_games, teams_list):
    n_teams = len(teams_list)
    t_idx = {t: i for i, t in enumerate(teams_list)}
    margin_vector = np.zeros(n_teams)
    games_matrix = np.zeros((n_teams, n_teams))
    games_played = np.zeros(n_teams)
    for _, row in df_season_games.iterrows():
        if row['WTeamID'] not in t_idx or row['LTeamID'] not in t_idx: continue
        w, l = t_idx[row['WTeamID']], t_idx[row['LTeamID']]
        margin = row['WScore'] - row['LScore']
        margin_vector[w] += margin
        margin_vector[l] -= margin
        games_matrix[w, l] += 1
        games_matrix[l, w] += 1
        games_played[w] += 1
        games_played[l] += 1
    avg_margin = np.divide(margin_vector, games_played, out=np.zeros_like(margin_vector), where=games_played!=0)
    A = np.zeros((n_teams, n_teams))
    for i in range(n_teams):
        A[i, i] = 1.0
        for j in range(n_teams):
            if i != j and games_played[i] > 0: A[i, j] = -(games_matrix[i, j] / games_played[i])
    A += np.eye(n_teams) * 0.05 
    # LinAlg Solve can be fragile, using a robust way
    try: srs_scores = np.linalg.solve(A, avg_margin)
    except: srs_scores = avg_margin # Rollback to margin if solve fails
    return {teams_list[i]: srs_scores[i] for i in range(n_teams)}

def build_advanced_ratings(gender='M'):
    reg_df = pd.read_csv(os.path.join(DATA_DIR, f'{gender}RegularSeasonCompactResults.csv'))
    reg_sorted = reg_df.sort_values(['Season', 'DayNum'])
    elo_df, _ = compute_advanced_elo(reg_sorted)
    w_elo = elo_df.groupby(['Season', 'WTeamID'])['W_Elo_Post'].last().reset_index().rename(columns={'WTeamID': 'TeamID', 'W_Elo_Post': 'Elo'})
    l_elo = elo_df.groupby(['Season', 'LTeamID'])['L_Elo_Post'].last().reset_index().rename(columns={'LTeamID': 'TeamID', 'L_Elo_Post': 'Elo'})
    team_elodf = pd.concat([w_elo, l_elo]).groupby(['Season', 'TeamID'])['Elo'].last().reset_index()
    srs_records = []
    for s in reg_df['Season'].unique():
        s_df = reg_df[reg_df['Season'] == s]
        teams = set(s_df['WTeamID']).union(set(s_df['LTeamID']))
        srs_dict = compute_srs(s_df, list(teams))
        for t, val in srs_dict.items(): srs_records.append({'Season': s, 'TeamID': t, 'SRS': val})
    team_srs = pd.DataFrame(srs_records)
    w_stats = reg_df.groupby(['Season', 'WTeamID']).agg(WScore=('WScore', 'mean'), LScore=('LScore', 'mean')).reset_index().rename(columns={'WTeamID':'TeamID'})
    l_stats = reg_df.groupby(['Season', 'LTeamID']).agg(LScore=('LScore', 'mean'), WScore=('WScore', 'mean')).reset_index().rename(columns={'LTeamID':'TeamID'})
    cmb = pd.concat([w_stats, l_stats]).groupby(['Season', 'TeamID']).mean().reset_index()
    cmb['Pyth_Expected'] = (cmb['WScore']**11.5) / ((cmb['WScore']**11.5) + (cmb['LScore']**11.5) + 1e-9)
    final_ratings = team_elodf.merge(team_srs, on=['Season', 'TeamID'], how='left').merge(cmb[['Season', 'TeamID', 'Pyth_Expected']], on=['Season', 'TeamID'], how='left')
    return final_ratings

# ==============================================================================
# 2. 1985년부터의 전체 데이터 로드 함수 (V14: 지능적 통합)
# ==============================================================================
def load_v14_data(gender='M'):
    print(f"🚀 {gender} 1985년~전 기간 데이터 로드 및 V8 피처 복원 중...")
    
    # 기초 토너먼트 데이터
    seeds = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneySeeds.csv'))
    tourney_results = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneyCompactResults.csv'))
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    
    tourney_results['T1'] = tourney_results.apply(lambda r: min(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['T2'] = tourney_results.apply(lambda r: max(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['Label'] = (tourney_results['T1'] == tourney_results['WTeamID']).astype(int)
    
    df = tourney_results[['Season', 'DayNum', 'T1', 'T2', 'Label']]
    
    # 시드 결합
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T1_SeedNum'})
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T2_SeedNum'})
    df['SeedNum_Diff'] = df['T1_SeedNum'] - df['T2_SeedNum']
    
    # 1. Advanced Ratings (Elo, SRS, Pyth) - 1985년부터 직접 계산
    adv_ratings = build_advanced_ratings(gender)
    for side in ['T1', 'T2']:
        df = pd.merge(df, adv_ratings, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
        df.rename(columns={c: f'{side}_{c}' for c in adv_ratings.columns if c not in ['Season', 'TeamID']}, inplace=True)
    for c in ['Elo', 'SRS', 'Pyth_Expected']:
        df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']
    
    # 2. 전처리된 V11/V8 핵심 피처 병합 (2003년 이후 위주이나 매칭되는 시즌만 사용)
    feature_files = glob.glob(os.path.join(PREP_DIR, f'*_{gender}.csv'))
    for file in feature_files:
        if 'base_matchup_features' in os.path.basename(file) or 'advanced_ratings' in os.path.basename(file): continue
        feat_df = pd.read_csv(file)
        if 'TeamID' in feat_df.columns:
            for side in ['T1', 'T2']:
                cols = ['Season', 'TeamID'] + [c for c in feat_df.columns if c not in ['Season', 'TeamID'] and f'{side}_{c}' not in df.columns]
                df = pd.merge(df, feat_df[cols], left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
                df.rename(columns={c: f'{side}_{c}' for c in feat_df.columns if c not in ['Season', 'TeamID']}, inplace=True)
            for c in feat_df.columns:
                if c not in ['Season', 'TeamID']:
                    if f'T1_{c}' in df.columns and f'T2_{c}' in df.columns:
                        df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']
    
    # 3. Massey Ordinals (Pre-2003 특수 처리)
    if gender == 'M':
        massey = pd.read_csv(os.path.join(DATA_DIR, 'MMasseyOrdinals.csv'))
        last_massey = massey[massey['RankingDayNum'] == 133].groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index().rename(columns={'OrdinalRank':'MOR'})
        df['Is_Pre_Massey'] = (df['Season'] < 2003).astype(int)
        for side in ['T1', 'T2']:
            df = pd.merge(df, last_massey, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            # 2003년 이전은 중립값 100위(MOR Diff 0 유도) 처리
            df['MOR'] = df['MOR'].fillna(100)
            df.rename(columns={'MOR': f'{side}_MOR'}, inplace=True)
        df['MORDiff'] = df['T1_MOR'] - df['T2_MOR']
    else:
        df['Is_Pre_Massey'] = 0
        df['MORDiff'] = 0
        
    return df

# ==============================================================================
# 3. 훈련 유틸리트 (V8 복구)
# ==============================================================================
def augment_data_v14(df, features, target='Label', noise_scale=0.03):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    df_swap[target] = 1 - df_swap[target]
    df_augmented = pd.concat([df, df_swap], ignore_index=True)
    numeric_feats = [f for f in features if f in df_augmented.columns]
    feat_stds = df_augmented[numeric_feats].std().values
    feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(len(df_augmented), len(numeric_feats)))
    df_augmented[numeric_feats] += (noise * feat_stds)
    return df_augmented

def get_strong_decay_weights(seasons_array, max_season, decay=0.55):
    # 80년대 데이터의 비중을 더 급격하게 낮춤 (현대 농구 데이터 강조)
    weights = decay ** (max_season - seasons_array)
    return weights / weights.mean()

def label_smoothing(y, val=0.05):
    return y * (1 - val) + 0.5 * val

# ==============================================================================
# 4. 메인 실행부
# ==============================================================================
print("🚀 V14: Top Performance Recovery (V8) + Intel-80s Integration", flush=True)

df_m = load_v14_data('M')
df_w = load_v14_data('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)

# Interaction 피처 복원 (V8)
df_train['Elo_x_SRS_Diff'] = df_train['Elo_Diff'] * df_train['SRS_Diff']
df_train['WinRate_x_Pyth_Diff'] = df_train['WinRate_Diff'] * df_train['Pyth_Expected_Diff']
df_train['Tourney_Efficiency_Diff'] = df_train['Past_Tourney_eFG%_Diff'] * (1 - df_train['Past_Tourney_TOV%_Diff'])

# V7에서 검증된 핵심 33개 피처 + 고유 피처
with open('selected_features_v7.json', 'r') as f:
    v7_features = json.load(f)

# V14 핵심 피처 세트 (V8 DNA)
features = v7_features + ['Elo_x_SRS_Diff', 'WinRate_x_Pyth_Diff', 'Tourney_Efficiency_Diff', 'Is_Pre_Massey']
features = [f for f in features if f in df_train.columns]

print(f"📊 최종 V14 피처 수: {len(features)}", flush=True)
print(f"📊 학습 데이터 총 행 수: {len(df_train)} (1985~2025)", flush=True)

# 2013-2025 풀 시즌 검증
full_val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
final_scores = []

# CatBoost V8 기반 최적 파라미터 (일부 조정)
params = {
    'iterations': 1500, 'learning_rate': 0.02, 'depth': 5,
    'l2_leaf_reg': 5.0, 'subsample': 0.75, 'colsample_bylevel': 0.7,
    'random_seed': SEED, 'loss_function': 'RMSE', 'bootstrap_type': 'Bernoulli',
    'verbose': False, 'thread_count': N_JOBS
}

print("\n🚀 V14 풀 시즌 평가 시작 (강력한 시간 감쇠 및 80년대 데이터 활용)...", flush=True)
print("| Year | Brier Score | 비고 |", flush=True)

for val_year in full_val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    df_tr = df_train.loc[train_mask, features + ['Label', 'Season']]
    # 데이터 증강 (Swap + Noise)
    df_tr_aug = augment_data_v14(df_tr, features, 'Label', noise_scale=0.03)
    
    # 현대 농구(최근 데이터)에 0.55 비중의 강력한 가중치 적용
    weights = get_strong_decay_weights(df_tr_aug['Season'], val_year)
    y_smooth = label_smoothing(df_tr_aug['Label'])
    
    model = CatBoostRegressor(**params)
    model.fit(df_tr_aug[features], y_smooth, sample_weight=weights)
    
    preds = np.clip(model.predict(df_train.loc[val_mask, features]), 0.0, 1.0)
    score = brier_score_loss(df_train.loc[val_mask, 'Label'], preds)
    final_scores.append(score)
    
    print(f"| {val_year} | {score:.4f} | {'최신 성능' if val_year >= 2023 else ''} |", flush=True)

print("-" * 60, flush=True)
print(f"🎯 최종 V14 (V8 DNA + 80s Decay) 평균 Brier Score: {np.mean(final_scores):.4f}", flush=True)
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 60, flush=True)
