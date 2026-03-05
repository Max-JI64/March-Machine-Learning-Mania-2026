import os
import glob
import time
import pandas as pd
import numpy as np
import optuna
import random
from sklearn.metrics import brier_score_loss
import xgboost as xgb
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# 파트 I: Advanced Ratings (Elo, SRS, Massey, Pyth, H2H 등) 함수 추가
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
            for t in elo_dict.keys():
                elo_dict[t] = 1500 * (1 - REVERSION) + elo_dict[t] * REVERSION
            current_season = season
            
        elo_w = elo_dict.get(w_team, 1500)
        elo_l = elo_dict.get(l_team, 1500)
        
        elo_w_adj = elo_w + HOME_ADV if w_loc == 'H' else (elo_w - HOME_ADV if w_loc == 'A' else elo_w)
        expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w_adj) / 400.0))
        
        margin_multiplier = np.log(min(abs(margin), 25) + 1) / np.log(26)
        update = K * margin_multiplier * (1.0 - expected_w)
        
        elo_dict[w_team] = elo_w + update
        elo_dict[l_team] = elo_l - update
        
        elo_records.append({
            'Season': season, 'DayNum': row['DayNum'],
            'WTeamID': w_team, 'LTeamID': l_team,
            'W_Elo_Post': elo_dict[w_team], 'L_Elo_Post': elo_dict[l_team]
        })
        
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
            if i != j and games_played[i] > 0:
                A[i, j] = -(games_matrix[i, j] / games_played[i])
                
    A += np.eye(n_teams) * 0.05 
    
    srs_scores = np.linalg.solve(A, avg_margin)
    return {teams_list[i]: srs_scores[i] for i in range(n_teams)}

def build_advanced_ratings(gender='M', data_dir='../Data/provided'):
    """
    모든 시즌 데이터를 불러와 누적 I 파트 피처(Elo, SRS, Pyth)를 DataFrame 형태로 반환합니다.
    """
    reg_df = pd.read_csv(os.path.join(data_dir, f'{gender}RegularSeasonCompactResults.csv'))
    
    # 1. Elo Rating
    reg_sorted = reg_df.sort_values(['Season', 'DayNum'])
    elo_df, _ = compute_advanced_elo(reg_sorted)
    
    # 시즌별 팀 엘로 마지막 값 추론
    w_elo = elo_df.groupby(['Season', 'WTeamID'])['W_Elo_Post'].last().reset_index().rename(columns={'WTeamID': 'TeamID', 'W_Elo_Post': 'Elo'})
    l_elo = elo_df.groupby(['Season', 'LTeamID'])['L_Elo_Post'].last().reset_index().rename(columns={'LTeamID': 'TeamID', 'L_Elo_Post': 'Elo'})
    team_elodf = pd.concat([w_elo, l_elo]).groupby(['Season', 'TeamID'])['Elo'].last().reset_index()

    # 2. SRS
    srs_records = []
    for s in reg_df['Season'].unique():
        s_df = reg_df[reg_df['Season'] == s]
        teams = set(s_df['WTeamID']).union(set(s_df['LTeamID']))
        srs_dict = compute_srs(s_df, list(teams))
        for t, val in srs_dict.items():
            srs_records.append({'Season': s, 'TeamID': t, 'SRS': val})
            
    team_srs = pd.DataFrame(srs_records)
    
    # 3. Pythagorean
    w_stats = reg_df.groupby(['Season', 'WTeamID']).agg(WScore=('WScore', 'mean'), LScore=('LScore', 'mean')).reset_index().rename(columns={'WTeamID':'TeamID'})
    l_stats = reg_df.groupby(['Season', 'LTeamID']).agg(LScore=('LScore', 'mean'), WScore=('WScore', 'mean')).reset_index().rename(columns={'LTeamID':'TeamID'})
    cmb = pd.concat([w_stats, l_stats]).groupby(['Season', 'TeamID']).mean().reset_index()
    cmb['Pyth_Expected'] = (cmb['WScore']**11.5) / ((cmb['WScore']**11.5) + (cmb['LScore']**11.5) + 1e-9)
    
    # 최종 머지
    final_ratings = team_elodf.merge(team_srs, on=['Season', 'TeamID'], how='left')\
                              .merge(cmb[['Season', 'TeamID', 'Pyth_Expected']], on=['Season', 'TeamID'], how='left')
    return final_ratings

# ==============================================================================
# 0. 재현성을 위한 시드 설정 (Random Seeding)
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

SEED = 42
set_seed(SEED)

START_TIME = time.time()
print(f"✅ 재현성 시드 고정: {SEED}")
print(f"⏱️ 분석 시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================================
# 1. 디바이스 및 코어 수 설정 (로컬 맥북 CPU용)
# ==============================================================================
DEVICE = 'cpu'
N_JOBS = os.cpu_count()
print(f"✅ 현재 설정된 학습 디바이스: {DEVICE.upper()}, CPU 코어 수: {N_JOBS}")

# ==============================================================================
# 2. 전처리된 데이터 로드 및 병합 (새로운 I 파트 피처 포함)
# ==============================================================================
def load_and_merge_features(gender='M', data_dir='../Data/preprocessed'):
    print(f"[{gender}] 데이터 로딩 및 병합 시작...")
    
    base_file = os.path.join(data_dir, f'base_matchup_features_{gender}.csv')
    df_main = pd.read_csv(base_file)
    
    feature_files = glob.glob(os.path.join(data_dir, f'*_{gender}.csv'))
    if gender == 'M':
        feature_files.append(os.path.join(data_dir, 'base_massey_features.csv'))
        
    for file in feature_files:
        if 'base_matchup_features' in os.path.basename(file) or 'advanced_ratings' in os.path.basename(file):
            continue
            
        feat_df = pd.read_csv(file)
        
        if 'TeamID' in feat_df.columns:
            # T1 병합
            cols_t1 = ['Season', 'TeamID'] + [
                col for col in feat_df.columns 
                if col not in ['Season', 'TeamID'] and f'T1_{col}' not in df_main.columns
            ]
            df_main = pd.merge(df_main, feat_df[cols_t1], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left')
            df_main.drop('TeamID', axis=1, inplace=True)
            df_main.rename(columns={col: f'T1_{col}' for col in cols_t1 if col not in ['Season', 'TeamID']}, inplace=True)
            
            # T2 병합
            cols_t2 = ['Season', 'TeamID'] + [
                col for col in feat_df.columns 
                if col not in ['Season', 'TeamID'] and f'T2_{col}' not in df_main.columns
            ]
            df_main = pd.merge(df_main, feat_df[cols_t2], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left')
            df_main.drop('TeamID', axis=1, inplace=True)
            df_main.rename(columns={col: f'T2_{col}' for col in cols_t2 if col not in ['Season', 'TeamID']}, inplace=True)
            
            # Diff 생성
            for col in feat_df.columns:
                if col not in ['Season', 'TeamID']:
                    diff_col = f'{col}_Diff'
                    if diff_col not in df_main.columns:
                        if f'T1_{col}' in df_main.columns and f'T2_{col}' in df_main.columns:
                            try:
                                df_main[diff_col] = df_main[f'T1_{col}'] - df_main[f'T2_{col}']
                            except:
                                pass
    
    # 여기서 I 파트(Advances Ratings)를 동적으로 병합
    adv_ratings = build_advanced_ratings(gender)
    
    cols_t1 = ['Season', 'TeamID'] + [c for c in adv_ratings.columns if c not in ['Season', 'TeamID']]
    df_main = pd.merge(df_main, adv_ratings[cols_t1], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left')
    df_main.rename(columns={col: f'T1_{col}' for col in cols_t1 if col not in ['Season', 'TeamID']}, inplace=True)
    if 'TeamID' in df_main.columns:
        df_main.drop('TeamID', axis=1, inplace=True)
    
    cols_t2 = ['Season', 'TeamID'] + [c for c in adv_ratings.columns if c not in ['Season', 'TeamID']]
    df_main = pd.merge(df_main, adv_ratings[cols_t2], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left')
    df_main.rename(columns={col: f'T2_{col}' for col in cols_t2 if col not in ['Season', 'TeamID']}, inplace=True)
    if 'TeamID' in df_main.columns:
        df_main.drop('TeamID', axis=1, inplace=True)
    
    for col in adv_ratings.columns:
        if col not in ['Season', 'TeamID']:
             diff_col = f'{col}_Diff'
             df_main[diff_col] = df_main[f'T1_{col}'] - df_main[f'T2_{col}']

    return df_main


# ==============================================================================
# 3. J파트: 새로운 데이터 증강 룰 포함 (Data Augmentation)
# ==============================================================================
def augment_data(df, features, target='Label', noise_scale=0.03):
    """
    1. 대칭 스왑 (T1↔T2 반전) 
    2. 가우시안 노이즈 추가
    """
    df_swap = df.copy()
    
    # T1_, T2_ 스왑
    t1_cols = [c for c in df.columns if c.startswith('T1_')]
    t2_cols = [c for c in df.columns if c.startswith('T2_')]
    for t1c, t2c in zip(sorted(t1_cols), sorted(t2_cols)):
        df_swap[t1c], df_swap[t2c] = df[t2c].values, df[t1c].values
    
    if 'T1' in df.columns and 'T2' in df.columns:
        df_swap['T1'], df_swap['T2'] = df['T2'].values, df['T1'].values
    
    # Diff 반전
    diff_cols = [c for c in df.columns if (c.endswith('_Diff') or c.endswith('Diff')) and c in features]
    for dc in diff_cols:
        df_swap[dc] = -df_swap[dc]
    
    df_swap[target] = 1 - df_swap[target]
    df_swap['Is_Augmented'] = 1
    df['Is_Augmented'] = 0
    
    df_augmented = pd.concat([df, df_swap], ignore_index=True)
    
    # 노이즈 추가
    aug_mask = df_augmented['Is_Augmented'] == 1
    numeric_feats = [f for f in features if f in df_augmented.columns and f != 'Is_Augmented']
    
    feat_stds = df_augmented.loc[~aug_mask, numeric_feats].std().values
    feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    
    noise = np.random.normal(0, noise_scale, size=(aug_mask.sum(), len(numeric_feats)))
    df_augmented.loc[aug_mask, numeric_feats] += (noise * feat_stds)
    
    return df_augmented

def get_recency_sample_weights(seasons_array, max_season=None, decay=0.60):
    """
    3. 타임라인 가중치 (Recency Weights) 부여 : 오래전 대회일수록 트리의 가중치 하락
    """
    if max_season is None:
        max_season = seasons_array.max()
    weights = decay ** (max_season - seasons_array)
    weights = weights / weights.mean()
    return weights

def label_smoothing(y_target, smoothing_val=0.05):
    """
    4. Label Smoothing : 극단 확률로 인한 Brier Score 페널티 예방용 타겟 보정
    """
    return y_target * (1 - smoothing_val) + 0.5 * smoothing_val

# ==============================================================================
# 4. 데이터 로드 및 전처리
# ==============================================================================
df_m = load_and_merge_features('M')
df_w = load_and_merge_features('W')

common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)

target = 'Label'
drop_cols = ['Season', 'T1', 'T2', 'Label', 'Is_Augmented'] + [c for c in df_train.columns if c.startswith('T1_') or c.startswith('T2_')]
features = sorted([c for c in df_train.columns if c not in drop_cols])

val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2018 and s != 2020])
min_season = int(df_train['Season'].min())

print(f"\n✅ 사용된 독립 피처 수: {len(features)}개")
print(f"✅ 학습 데이터 기반: {min_season} ~ 2025")


# ==============================================================================
# 5. 하이퍼파라미터 최적화 (새로운 전처리, 증강 포함! -> 1차 시기 재시작)
# ==============================================================================
def objective(trial):
    # 1차 시기(넓은 탐색 공간) 룰 적용
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'logloss',
        'random_state': SEED,
        'nthread': N_JOBS
    }
    
    cv_briers = []
    
    # 튜닝 속도를 고려해 최근 3시즌만 평가하되, J파트 증강은 반드시 적용 (교훈 반영)
    for val_year in val_seasons[-3:]:
        train_mask = (df_train['Season'] < val_year)
        val_mask = (df_train['Season'] == val_year)
        
        X_tr, y_tr = df_train.loc[train_mask, features].copy(), df_train.loc[train_mask, target].copy()
        X_val, y_val = df_train.loc[val_mask, features].copy(), df_train.loc[val_mask, target].copy()
        
        if len(X_val) == 0: continue

        # 🚨 Optuna 단계에서도 증강 적용!
        train_temp = pd.concat([X_tr, y_tr], axis=1)
        train_temp['Season'] = df_train.loc[train_mask, 'Season']
        train_aug = augment_data(train_temp, features, target)
        
        X_tr_aug, y_tr_aug = train_aug[features], train_aug[target]
        seasons_aug = train_aug['Season']
        
        # J 파트: Sample Weights 및 Label Smoothing 반영
        weights_aug = get_recency_sample_weights(seasons_aug, max_season=val_year)
        y_tr_smooth = label_smoothing(y_tr_aug)

        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(X_tr_aug, y_tr_smooth, sample_weight=weights_aug, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        cv_briers.append(brier_score_loss(y_val, preds))
        
    return np.mean(cv_briers)

print("\n🚀 1단계: 하이퍼파라미터 최적화 (XGBoost Local, 넓은 공간, 증강 포함 20 trials)...")
# 주의: 증강데이터가 포함되므로 로컬 CPU에서는 20회도 약간 오래 걸릴 수 있습니다.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

best_params = study.best_params
best_params.update({
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'logloss', 
    'nthread': N_JOBS,
    'random_state': SEED
})
print('\n✅ [최적 파라미터]', best_params)


# ==============================================================================
# 6. 최종 Time-based CV 평가 (J파트 증강 전면 적용)
# ==============================================================================
print("\n🚀 2단계: 최적 파라미터로 모든 연도 최종 성능 평가 시작...")
final_briers = []

for val_year in val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    X_tr = df_train.loc[train_mask, features].copy()
    y_tr = df_train.loc[train_mask, target].copy()
    X_val = df_train.loc[val_mask, features].copy()
    y_val = df_train.loc[val_mask, target].copy()
    
    if len(X_val) == 0: continue

    # J파트: 최종 학습에도 증강 적용
    train_temp = pd.concat([X_tr, y_tr], axis=1)
    train_temp['Season'] = df_train.loc[train_mask, 'Season']
    train_aug = augment_data(train_temp, features, target)
    
    X_tr_aug = train_aug[features]
    y_tr_aug = train_aug[target]
    seasons_aug = train_aug['Season']
    
    weights_aug = get_recency_sample_weights(seasons_aug, max_season=val_year)
    y_tr_smooth = label_smoothing(y_tr_aug)
        
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_tr_aug, y_tr_smooth, sample_weight=weights_aug, verbose=False)
    
    preds = model.predict(X_val)
    brier = brier_score_loss(y_val, preds)
    final_briers.append(brier)
    print(f"-> Season {val_year} 검증 Brier Score: {brier:.4f} (증강학습: {len(X_tr_aug)}행)")

elapsed = time.time() - START_TIME
print("-" * 60)
print(f"🎯 전체 연도 최종 평균 Brier Score: {np.mean(final_briers):.4f}")
print(f"⏱️ 총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.0f}초)")
print(f"⏱️ 분석 종료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 60)
