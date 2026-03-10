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
import optuna

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

# ==============================================================================
# 데이터 로드 및 병합 함수 (V11 전용)
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
    srs_scores = np.linalg.solve(A, avg_margin)
    return {teams_list[i]: srs_scores[i] for i in range(n_teams)}

def build_advanced_ratings(gender='M', data_dir='../Data/provided'):
    reg_df = pd.read_csv(os.path.join(data_dir, f'{gender}RegularSeasonCompactResults.csv'))
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

def load_and_merge_features(gender='M', data_dir='../Data/preprocessed'):
    base_file = os.path.join(data_dir, f'base_matchup_features_{gender}.csv')
    df_main = pd.read_csv(base_file)
    feature_files = glob.glob(os.path.join(data_dir, f'*_{gender}.csv'))
    if gender == 'M': feature_files.append(os.path.join(data_dir, 'base_massey_features.csv'))
    for file in feature_files:
        if 'base_matchup_features' in os.path.basename(file) or 'advanced_ratings' in os.path.basename(file): continue
        feat_df = pd.read_csv(file)
        if 'TeamID' in feat_df.columns:
            for side in ['T1', 'T2']:
                cols = ['Season', 'TeamID'] + [c for c in feat_df.columns if c not in ['Season', 'TeamID'] and f'{side}_{c}' not in df_main.columns]
                df_main = pd.merge(df_main, feat_df[cols], left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
                df_main.rename(columns={c: f'{side}_{c}' for c in cols if c not in ['Season', 'TeamID']}, inplace=True)
            for c in feat_df.columns:
                if c not in ['Season', 'TeamID']:
                    if f'T1_{c}' in df_main.columns and f'T2_{c}' in df_main.columns:
                        df_main[f'{c}_Diff'] = df_main[f'T1_{c}'] - df_main[f'T2_{c}']
    adv_ratings = build_advanced_ratings(gender)
    for side in ['T1', 'T2']:
        cols = ['Season', 'TeamID'] + [c for c in adv_ratings.columns if c not in ['Season', 'TeamID']]
        df_main = pd.merge(df_main, adv_ratings[cols], left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
        df_main.rename(columns={c: f'{side}_{c}' for c in cols if c not in ['Season', 'TeamID']}, inplace=True)
    for c in adv_ratings.columns:
        if c not in ['Season', 'TeamID']: df_main[f'{c}_Diff'] = df_main[f'T1_{c}'] - df_main[f'T2_{c}']
    return df_main

def augment_data(df, features, target='Label', noise_scale=0.03):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    df_swap[target] = 1 - df_swap[target]
    df_augmented = pd.concat([df, df_swap], ignore_index=True)
    numeric_feats = [f for f in features if f in df_augmented.columns]
    feat_stds = df_augmented[numeric_feats].std().values
    noise = np.random.normal(0, noise_scale, size=(len(df_augmented), len(numeric_feats)))
    df_augmented[numeric_feats] += (noise * feat_stds)
    return df_augmented

def get_recency_sample_weights(seasons_array, max_season=None, decay=0.60):
    if max_season is None: max_season = seasons_array.max()
    weights = decay ** (max_season - seasons_array)
    return weights / weights.mean()

def label_smoothing(y_target, smoothing_val=0.05):
    return y_target * (1 - smoothing_val) + 0.5 * smoothing_val

# ==============================================================================
# 메인 실행부
# ==============================================================================
print("🚀 V11: Automatic Feature Selection & Training (Step 2 Implementation)", flush=True)

# 1. 고도화된 모멘텀 피처가 포함된 데이터 로드
df_m = load_and_merge_features('M')
df_w = load_and_merge_features('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)
target = 'Label'

# 2. V8 Interaction 피처 생성
if 'Elo_Diff' in df_train.columns and 'SRS_Diff' in df_train.columns:
    df_train['Elo_x_SRS_Diff'] = df_train['Elo_Diff'] * df_train['SRS_Diff']
if 'WinRate_Diff' in df_train.columns and 'Pyth_Expected_Diff' in df_train.columns:
    df_train['WinRate_x_Pyth_Diff'] = df_train['WinRate_Diff'] * df_train['Pyth_Expected_Diff']
if 'Tourney_Efficiency_Diff' not in df_train.columns: # V8에서 썼던 로직
    if 'Past_Tourney_eFG%_Diff' in df_train.columns and 'Past_Tourney_TOV%_Diff' in df_train.columns:
        df_train['Tourney_Efficiency_Diff'] = df_train['Past_Tourney_eFG%_Diff'] * (1 - df_train['Past_Tourney_TOV%_Diff'])

# 모든 가용 피처 후보군 추출 (Diff 계열 + 신규 모멘텀)
drop_cols = ['Season', 'T1', 'T2', 'Label'] + [c for c in df_train.columns if c.startswith('T1_') or c.startswith('T2_')]
all_features = sorted([c for c in df_train.columns if c not in drop_cols and (c.endswith('_Diff') or c == 'Active_Streak_Diff')])

# ==============================================================================
# 3. 자동 변수 선택 (Using CatBoost Importance)
# ==============================================================================
print(f"📊 1차 전체 피처 학습 및 자동 선택 진행 중... (총 {len(all_features)}개 피처)", flush=True)

# 최신 5개년 데이터로 중요도 측정
importance_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2019 and s != 2020])
avg_importance = pd.Series(0, index=all_features)

for s in importance_seasons:
    train_mask = (df_train['Season'] < s)
    X_imp, y_imp = df_train.loc[train_mask, all_features], df_train.loc[train_mask, target]
    
    # 가벼운 학습으로 중요도 추출
    model_imp = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=5, verbose=False, thread_count=N_JOBS, random_seed=SEED)
    model_imp.fit(X_imp, y_imp)
    avg_importance += model_imp.get_feature_importance()

avg_importance /= len(importance_seasons)
selected_features = avg_importance.sort_values(ascending=False).head(35).index.tolist()

print(f"✅ 자동 변수 선택 완료: 상위 {len(selected_features)}개 피처 선별", flush=True)
print(f"📌 주요 피처: {selected_features[:10]}")

with open('selected_features_v11.json', 'w') as f:
    json.dump(selected_features, f)

# ==============================================================================
# 4. 선택된 피처로 최종 검증 및 학습 (2013-2025)
# ==============================================================================
cat_params = {
    'iterations': 1200, 'learning_rate': 0.02, 'depth': 6,
    'l2_leaf_reg': 6.0, 'subsample': 0.8, 'colsample_bylevel': 0.7,
    'random_seed': SEED, 'loss_function': 'RMSE', 'bootstrap_type': 'Bernoulli',
    'verbose': False, 'thread_count': N_JOBS
}

full_val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
final_scores = []

print("\n🚀 V11 최종 2013-2025 풀 시즌 평가 진행...", flush=True)
for val_year in full_val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    # 증강 및 가중치 적용
    train_temp = df_train.loc[train_mask, selected_features + [target, 'Season']]
    train_aug = augment_data(train_temp, selected_features, target, noise_scale=0.025)
    weights_aug = get_recency_sample_weights(train_aug['Season'], max_season=val_year)
    y_tr_smooth = label_smoothing(train_aug[target])
    
    model = CatBoostRegressor(**cat_params)
    model.fit(train_aug[selected_features], y_tr_smooth, sample_weight=weights_aug)
    
    preds = np.clip(model.predict(df_train.loc[val_mask, selected_features]), 0, 1)
    score = brier_score_loss(df_train.loc[val_mask, target], preds)
    final_scores.append(score)
    print(f"| {val_year} | {score:.4f} |", flush=True)

print("-" * 60, flush=True)
print(f"🎯 최종 V11 (자동 변수 선택) 평균 Brier Score: {np.mean(final_scores):.4f}", flush=True)
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 60, flush=True)
