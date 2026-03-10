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
# 1. 시뮬레이션 데이터 로드 함수 (V14 기반)
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
        elo_w += update
        elo_l -= update
        elo_dict[w_team] = elo_w
        elo_dict[l_team] = elo_l
        elo_records.append({'Season': season, 'DayNum': row['DayNum'], 'WTeamID': w_team, 'LTeamID': l_team, 'W_Elo_Post': elo_dict[w_team], 'L_Elo_Post': elo_dict[l_team]})
    return pd.DataFrame(elo_records)

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
    try: srs_scores = np.linalg.solve(A, avg_margin)
    except: srs_scores = avg_margin 
    return {teams_list[i]: srs_scores[i] for i in range(n_teams)}

def build_advanced_ratings(gender='M'):
    reg_df = pd.read_csv(os.path.join(DATA_DIR, f'{gender}RegularSeasonCompactResults.csv'))
    elo_df = compute_advanced_elo(reg_df.sort_values(['Season', 'DayNum']))
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
    return team_elodf.merge(team_srs, on=['Season', 'TeamID'], how='left').merge(cmb[['Season', 'TeamID', 'Pyth_Expected']], on=['Season', 'TeamID'], how='left')

def load_v15_data(gender='M'):
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
    adv_ratings = build_advanced_ratings(gender)
    for side in ['T1', 'T2']:
        df = pd.merge(df, adv_ratings, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
        df.rename(columns={c: f'{side}_{c}' for c in adv_ratings.columns if c not in ['Season', 'TeamID']}, inplace=True)
    for c in ['Elo', 'SRS', 'Pyth_Expected']: df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']
    feature_files = glob.glob(os.path.join(PREP_DIR, f'*_{gender}.csv'))
    for file in feature_files:
        if 'base_matchup' in file or 'advanced_ratings' in file: continue
        feat_df = pd.read_csv(file)
        if 'TeamID' in feat_df.columns:
            for side in ['T1', 'T2']:
                cols = ['Season', 'TeamID'] + [c for c in feat_df.columns if c not in ['Season', 'TeamID'] and f'{side}_{c}' not in df.columns]
                df = pd.merge(df, feat_df[cols], left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
                df.rename(columns={c: f'{side}_{c}' for c in feat_df.columns if c not in ['Season', 'TeamID']}, inplace=True)
            for c in feat_df.columns:
                if c not in ['Season', 'TeamID'] and f'T1_{c}' in df.columns and f'T2_{c}' in df.columns: df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']
    if gender == 'M':
        massey = pd.read_csv(os.path.join(DATA_DIR, 'MMasseyOrdinals.csv'))
        last_massey = massey[massey['RankingDayNum'] == 133].groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index().rename(columns={'OrdinalRank':'MOR'})
        df['Is_Pre_Massey'] = (df['Season'] < 2003).astype(int)
        for side in ['T1', 'T2']:
            df = pd.merge(df, last_massey, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            df['MOR'] = df['MOR'].fillna(100)
            df.rename(columns={'MOR': f'{side}_MOR'}, inplace=True)
        df['MORDiff'] = df['T1_MOR'] - df['T2_MOR']
    else: df['Is_Pre_Massey'], df['MORDiff'] = 0, 0
    return df

# ==============================================================================
# 2. Optuna 튜닝 설정
# ==============================================================================
print("🚀 V15: CatBoost Deep Tuning (Hyperparams + Decay)", flush=True)

df_m = load_v15_data('M')
df_w = load_v15_data('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)

# Interaction 피처
df_train['Elo_x_SRS_Diff'] = df_train['Elo_Diff'] * df_train['SRS_Diff']
df_train['WinRate_x_Pyth_Diff'] = df_train['WinRate_Diff'] * df_train['Pyth_Expected_Diff']
df_train['Tourney_Efficiency_Diff'] = df_train['Past_Tourney_eFG%_Diff'] * (1 - df_train['Past_Tourney_TOV%_Diff'])

with open('selected_features_v7.json', 'r') as f:
    v7_features = json.load(f)
features = [f for f in v7_features + ['Elo_x_SRS_Diff', 'WinRate_x_Pyth_Diff', 'Tourney_Efficiency_Diff', 'Is_Pre_Massey'] if f in df_train.columns]

def augment_v15(df, features, target='Label', noise_scale=0.03):
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

def objective(trial):
    # 탐색 범위 파라미터
    decay = trial.suggest_float('decay', 0.45, 0.75)
    noise_scale = trial.suggest_float('noise_scale', 0.015, 0.045)
    
    cb_params = {
        'iterations': trial.suggest_int('iterations', 800, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.04, log=True),
        'depth': trial.suggest_int('depth', 4, 7),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 15.0),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.9),
        'random_seed': SEED,
        'loss_function': 'RMSE', # Brier Score 최적화에 직결되는 RMSE 사용
        'bootstrap_type': 'Bernoulli',
        'verbose': False,
        'thread_count': N_JOBS
    }
    
    # 평가용 연차 (수행 가능한 시간 내에서 최대한 확보)
    eval_years = [2018, 2019, 2021, 2022, 2023, 2024, 2025]
    scores = []
    
    for val_year in eval_years:
        train_mask = (df_train['Season'] < val_year)
        val_mask = (df_train['Season'] == val_year)
        
        df_tr = df_train.loc[train_mask, features + ['Label', 'Season']]
        df_tr_aug = augment_v15(df_tr, features, 'Label', noise_scale=noise_scale)
        
        weights = decay ** (val_year - df_tr_aug['Season'])
        weights = weights / weights.mean()
        
        model = CatBoostRegressor(**cb_params)
        model.fit(df_tr_aug[features], df_tr_aug['Label'], sample_weight=weights)
        
        preds = np.clip(model.predict(df_train.loc[val_mask, features]), 0.0, 1.0)
        scores.append(brier_score_loss(df_train.loc[val_mask, 'Label'], preds))
        
    return np.mean(scores)

# ==============================================================================
# 3. 최적화 실행
# ==============================================================================
print("\n🚀 Optuna 튜닝 시작 (50 Trials)...", flush=True)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("\n" + "="*60, flush=True)
print(f"🎯 Best Brier Score found: {study.best_value:.5f}", flush=True)
print("🎯 Best Parameters:", flush=True)
for k, v in study.best_params.items():
    print(f"   - {k}: {v}", flush=True)

# 4. 베스트 파라미터로 최종 전체 평가 (2013-2025)
print("\n🚀 베스트 조합으로 전 기간(2013-2025) 최종 평가...", flush=True)
best_p = study.best_params
best_noise = best_p.pop('noise_scale')
best_decay = best_p.pop('decay')
final_cb_params = {**best_p, 'random_seed': SEED, 'loss_function': 'RMSE', 'verbose': False, 'thread_count': N_JOBS}

full_val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
final_results = []

for val_year in full_val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    df_tr_aug = augment_v15(df_train.loc[train_mask, features + ['Label', 'Season']], features, 'Label', noise_scale=best_noise)
    weights = best_decay ** (val_year - df_tr_aug['Season'])
    weights /= weights.mean()
    
    model = CatBoostRegressor(**final_cb_params)
    model.fit(df_tr_aug[features], df_tr_aug['Label'], sample_weight=weights)
    preds = np.clip(model.predict(df_train.loc[val_mask, features]), 0.0, 1.0)
    score = brier_score_loss(df_train.loc[val_mask, 'Label'], preds)
    final_results.append(score)
    print(f"| {val_year} | {score:.4f} |", flush=True)

print("-" * 60, flush=True)
print(f"🎯 최종 V15 (최적화 완료) 평균 Brier Score: {np.mean(final_results):.4f}", flush=True)
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 60, flush=True)
