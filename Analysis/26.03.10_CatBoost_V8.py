import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
from sklearn.metrics import brier_score_loss
from catboost import CatBoostRegressor
import optuna
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 파트 I: Advanced Ratings 함수
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

# ==============================================================================
# 0. 증강 함수 및 기본 세팅
# ==============================================================================
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)
START_TIME = time.time()
N_JOBS = min(os.cpu_count(), 64)

# 토큰 낭비를 막기 위한 터미널 출력 최소화 안내
print("🚀 V8: Feature Engineering & CatBoost Deep Tuning (Token Optimized Run)")
print(f"👉 사용 스레드 수: {N_JOBS}")
print("👉 Optuna 탐색 중 CatBoost의 내부 평가지표 출력(수천 줄)은 숨겨집니다. (verbose=False)")

def augment_data(df, features, target='Label', noise_scale=0.03):
    df_swap = df.copy()
    t1_cols = sorted([c for c in df.columns if c.startswith('T1_')])
    t2_cols = sorted([c for c in df.columns if c.startswith('T2_')])
    for t1c, t2c in zip(t1_cols, t2_cols): df_swap[t1c], df_swap[t2c] = df[t2c].values, df[t1c].values
    if 'T1' in df.columns and 'T2' in df.columns: df_swap['T1'], df_swap['T2'] = df['T2'].values, df['T1'].values
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    df_swap[target] = 1 - df_swap[target]
    df_swap['Is_Augmented'] = 1
    df['Is_Augmented'] = 0
    df_augmented = pd.concat([df, df_swap], ignore_index=True)
    aug_mask = df_augmented['Is_Augmented'] == 1
    numeric_feats = [f for f in features if f in df_augmented.columns and f != 'Is_Augmented']
    feat_stds = df_augmented.loc[~aug_mask, numeric_feats].std().values
    feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(aug_mask.sum(), len(numeric_feats)))
    df_augmented.loc[aug_mask, numeric_feats] += (noise * feat_stds)
    return df_augmented

def get_recency_sample_weights(seasons_array, max_season=None, decay=0.60):
    if max_season is None: max_season = seasons_array.max()
    weights = decay ** (max_season - seasons_array)
    return weights / weights.mean()

def label_smoothing(y_target, smoothing_val=0.05):
    return y_target * (1 - smoothing_val) + 0.5 * smoothing_val

# 데이터 로드
df_m = load_and_merge_features('M')
df_w = load_and_merge_features('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)
target = 'Label'

# ==============================================================================
# 2. V8: Feature Engineering (파생 변수 생성)
# ==============================================================================
# V7에서 추출된 Top Features 불러오기
try:
    with open('selected_features_v7.json', 'r') as f:
        base_features = json.load(f)
    print(f"✅ V7 가지치기 완료 피처 세트 불러옴: {len(base_features)}개")
except FileNotFoundError:
    print("⚠️ selected_features_v7.json 파일을 찾을 수 없어 전체 생성합니다.")
    drop_cols = ['Season', 'T1', 'T2', 'Label', 'Is_Augmented'] + [c for c in df_train.columns if c.startswith('T1_') or c.startswith('T2_')]
    base_features = sorted([c for c in df_train.columns if c not in drop_cols and c.endswith('_Diff')])

# 상호작용 피처 생성 (Interaction Features)
new_features = []
# 1. Elo와 SRS 조합 (가장 중요한 두 지표의 시너지)
if 'Elo_Diff' in df_train.columns and 'SRS_Diff' in df_train.columns:
    df_train['Elo_x_SRS_Diff'] = df_train['Elo_Diff'] * df_train['SRS_Diff']
    new_features.append('Elo_x_SRS_Diff')

# 2. 승률과 피타고리안 기대 승률 조합
if 'WinRate_Diff' in df_train.columns and 'Pyth_Expected_Diff' in df_train.columns:
    df_train['WinRate_x_Pyth_Diff'] = df_train['WinRate_Diff'] * df_train['Pyth_Expected_Diff']
    new_features.append('WinRate_x_Pyth_Diff')

# 3. Offense & Defense 시너지 (득실차의 정밀 지표)
if 'ORB%_Diff' in df_train.columns and 'DRB%_Diff' in df_train.columns:
    df_train['Total_Rebound_Control_Diff'] = df_train['ORB%_Diff'] + df_train['DRB%_Diff']
    new_features.append('Total_Rebound_Control_Diff')

if 'Past_Tourney_eFG%_Diff' in df_train.columns and 'Past_Tourney_TOV%_Diff' in df_train.columns:
    # 슛팅 효율성 * 턴오버 안정성
    df_train['Tourney_Efficiency_Diff'] = df_train['Past_Tourney_eFG%_Diff'] * (1 - df_train['Past_Tourney_TOV%_Diff'])
    new_features.append('Tourney_Efficiency_Diff')

print(f"✅ 파생 변수 생성 완료: {new_features}")
features = base_features + new_features

val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2021])

# ==============================================================================
# 3. CatBoost Optuna Deep Tuning
# ==============================================================================
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 600, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.9),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'random_seed': SEED,
        'loss_function': 'RMSE', # Brier Score 대용
        'bootstrap_type': 'Bernoulli',
        'verbose': False,  # 터미널 토큰 최적화
        'thread_count': N_JOBS
    }
    
    noise_scale = trial.suggest_float('noise_scale', 0.01, 0.04)
    cv_scores = []
    
    # 평가 시간 단축을 위해 최신 4개년에 대해서만 튜닝 검증 (Rolling Holdout)
    tuning_seasons = [2022, 2023, 2024, 2025]
    
    for val_year in tuning_seasons:
        train_mask = (df_train['Season'] < val_year)
        val_mask = (df_train['Season'] == val_year)
        X_tr, y_tr = df_train.loc[train_mask, features].copy(), df_train.loc[train_mask, target].copy()
        X_val, y_val = df_train.loc[val_mask, features].copy(), df_train.loc[val_mask, target].copy()
        
        train_temp = pd.concat([X_tr, y_tr], axis=1)
        train_temp['Season'] = df_train.loc[train_mask, 'Season']
        train_aug = augment_data(train_temp, features, target, noise_scale=noise_scale)
        weights_aug = get_recency_sample_weights(train_aug['Season'], max_season=val_year)
        y_tr_smooth = label_smoothing(train_aug[target])
        
        model = CatBoostRegressor(**params)
        model.fit(train_aug[features], y_tr_smooth, sample_weight=weights_aug)
        p = np.clip(model.predict(X_val), 0.0, 1.0)
        cv_scores.append(brier_score_loss(y_val, p))
        
    return np.mean(cv_scores)

print("\n🚀 Optuna 하이퍼파라미터 딥튜닝 시작 (30 Trials)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, n_jobs=1) # 병렬 실행시 출력이 섞이는 것을 방지

print("\n" + "="*50)
print(f"🌟 Best Brier Score (Tuning Phase): {study.best_value:.5f}")
print("🌟 Best Params:")
for k, v in study.best_params.items():
    print(f"    '{k}': {v},")
print("="*50)

# ==============================================================================
# 4. 베스트 파라미터로 전체 연도(2013-2025) OOF 최종 검증
# ==============================================================================
print("\n🚀 베스트 파라미터로 최종 2013-2025 풀 시즌 평가 진행...")
best_params = study.best_params.copy()
best_noise = best_params.pop('noise_scale')
best_params.update({
    'random_seed': SEED,
    'loss_function': 'RMSE',
    'bootstrap_type': 'Bernoulli',
    'verbose': False,
    'thread_count': N_JOBS
})

full_val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
final_brier_per_year = []

for val_year in full_val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    X_tr, y_tr = df_train.loc[train_mask, features].copy(), df_train.loc[train_mask, target].copy()
    X_val, y_val = df_train.loc[val_mask, features].copy(), df_train.loc[val_mask, target].copy()
    
    train_temp = pd.concat([X_tr, y_tr], axis=1)
    train_temp['Season'] = df_train.loc[train_mask, 'Season']
    train_aug = augment_data(train_temp, features, target, noise_scale=best_noise)
    weights_aug = get_recency_sample_weights(train_aug['Season'], max_season=val_year)
    y_tr_smooth = label_smoothing(train_aug[target])
    
    model = CatBoostRegressor(**best_params)
    model.fit(train_aug[features], y_tr_smooth, sample_weight=weights_aug)
    p = np.clip(model.predict(X_val), 0.0, 1.0)
    
    score = brier_score_loss(y_val, p)
    final_brier_per_year.append(score)
    print(f"| {val_year} | {score:.4f} |")

print("-" * 60)
print(f"🎯 최종 CatBoost V8 (단일) 평균 Brier Score: {np.mean(final_brier_per_year):.4f}")
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분")
print("-" * 60)
