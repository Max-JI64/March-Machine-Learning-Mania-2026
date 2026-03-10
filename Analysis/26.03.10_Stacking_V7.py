import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ==============================================================================
# 파트 I: Advanced Ratings 함수 추가
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
        for t, val in srs_dict.items():
            srs_records.append({'Season': s, 'TeamID': t, 'SRS': val})
            
    team_srs = pd.DataFrame(srs_records)
    
    w_stats = reg_df.groupby(['Season', 'WTeamID']).agg(WScore=('WScore', 'mean'), LScore=('LScore', 'mean')).reset_index().rename(columns={'WTeamID':'TeamID'})
    l_stats = reg_df.groupby(['Season', 'LTeamID']).agg(LScore=('LScore', 'mean'), WScore=('WScore', 'mean')).reset_index().rename(columns={'LTeamID':'TeamID'})
    cmb = pd.concat([w_stats, l_stats]).groupby(['Season', 'TeamID']).mean().reset_index()
    cmb['Pyth_Expected'] = (cmb['WScore']**11.5) / ((cmb['WScore']**11.5) + (cmb['LScore']**11.5) + 1e-9)
    
    final_ratings = team_elodf.merge(team_srs, on=['Season', 'TeamID'], how='left')\
                              .merge(cmb[['Season', 'TeamID', 'Pyth_Expected']], on=['Season', 'TeamID'], how='left')
    return final_ratings

# ==============================================================================
# 0. 재현성 설정
# ==============================================================================
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)
START_TIME = time.time()

# ==============================================================================
# 1. 환경 설정
# ==============================================================================
N_JOBS = min(os.cpu_count(), 64)
print(f"✅ Stacking V7 (Logistic Level-2 | 3-Way Tree Base) - 사용 스레드 수: {N_JOBS}")

# ==============================================================================
# 2. 데이터 로드 및 병합
# ==============================================================================
def load_and_merge_features(gender='M', data_dir='../Data/preprocessed'):
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
        if c not in ['Season', 'TeamID']:
            df_main[f'{c}_Diff'] = df_main[f'T1_{c}'] - df_main[f'T2_{c}']

    return df_main

# ==============================================================================
# 3. 데이터 증강 및 가중치
# ==============================================================================
def augment_data(df, features, target='Label', noise_scale=0.03):
    df_swap = df.copy()
    t1_cols = sorted([c for c in df.columns if c.startswith('T1_')])
    t2_cols = sorted([c for c in df.columns if c.startswith('T2_')])
    for t1c, t2c in zip(t1_cols, t2_cols):
        df_swap[t1c], df_swap[t2c] = df[t2c].values, df[t1c].values
    
    if 'T1' in df.columns and 'T2' in df.columns:
        df_swap['T1'], df_swap['T2'] = df['T2'].values, df['T1'].values
    
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols:
        df_swap[dc] = -df_swap[dc]
    
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

# ==============================================================================
# 4. 데이터 로드 및 V7 프루닝 피처 필터링 적용
# ==============================================================================
df_m = load_and_merge_features('M')
df_w = load_and_merge_features('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)

target = 'Label'
drop_cols = ['Season', 'T1', 'T2', 'Label', 'Is_Augmented'] + [c for c in df_train.columns if c.startswith('T1_') or c.startswith('T2_')]
all_diff_features = sorted([c for c in df_train.columns if c not in drop_cols])

# 앞서 JSON으로 저장한 Top Features 불러오기
try:
    with open('selected_features_v7.json', 'r') as f:
        features = json.load(f)
    print(f"✅ V7 가지치기 완료 피처 세트 불러옴: {len(features)}개 (원래 {len(all_diff_features)}개에서 줄임)")
except FileNotFoundError:
    print("⚠️ selected_features_v7.json 파일을 찾을 수 없습니다. (전체 피처 사용합니다)")
    features = all_diff_features

val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])

# ==============================================================================
# 5. 베이스 모델 (Level-1) 최적 파라미터 로드
# ==============================================================================
xgb_params = {
    'n_estimators': 1113, 'learning_rate': 0.0305, 'max_depth': 4,
    'subsample': 0.7486, 'colsample_bytree': 0.6032, 'min_child_weight': 6,
    'gamma': 3.3322, 'alpha': 4.1672, 'lambda': 3.2610,
    'objective': 'binary:logistic', 'tree_method': 'hist', 'eval_metric': 'logloss', 
    'random_state': SEED, 'nthread': N_JOBS
}
xgb_noise = 0.0104

lgb_params = {
    'max_depth': 5, 'n_estimators': 952, 'learning_rate': 0.0234,
    'num_leaves': 28, 'subsample': 0.8219, 'colsample_bytree': 0.4009,
    'min_child_samples': 32, 'reg_alpha': 0.8095, 'reg_lambda': 5.9629,
    'objective': 'cross_entropy', 'metric': 'cross_entropy', 
    'n_jobs': N_JOBS, 'random_state': SEED, 'verbose': -1,
    'deterministic': True, 'force_row_wise': True
}
lgb_noise = 0.03

cat_params = {
    'iterations': 893, 'learning_rate': 0.0257, 'depth': 7,
    'l2_leaf_reg': 4.807, 'subsample': 0.675, 'colsample_bylevel': 0.708,
    'min_data_in_leaf': 32, 'random_seed': SEED, 'loss_function': 'RMSE',
    'verbose': False, 'thread_count': N_JOBS, 'bootstrap_type': 'Bernoulli'
}
cat_noise = 0.0209

# ==============================================================================
# 6. 블렌딩 검증 수행 및 Level-2(Meta Feature) 생성
# ==============================================================================
print("\n🚀 1단계: 신규 피처셋(33개) 기반 1차 트리 모델 학습 (2013-2025)...")

# 전체 OOF 저장을 위한 리스트
OOF_df_list = []

for val_year in val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    X_tr, y_tr = df_train.loc[train_mask, features].copy(), df_train.loc[train_mask, target].copy()
    X_val, y_val = df_train.loc[val_mask, features].copy(), df_train.loc[val_mask, target].copy()

    # --- XGBoost ---
    train_temp_xgb = pd.concat([X_tr, y_tr], axis=1)
    train_temp_xgb['Season'] = df_train.loc[train_mask, 'Season']
    train_aug_xgb = augment_data(train_temp_xgb, features, target, noise_scale=xgb_noise)
    weights_aug_xgb = get_recency_sample_weights(train_aug_xgb['Season'], max_season=val_year)
    y_tr_smooth_xgb = label_smoothing(train_aug_xgb[target])
    
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(train_aug_xgb[features], y_tr_smooth_xgb, sample_weight=weights_aug_xgb, verbose=False)
    p_xgb = model_xgb.predict(X_val)

    # --- LightGBM ---
    train_temp_lgb = pd.concat([X_tr, y_tr], axis=1)
    train_temp_lgb['Season'] = df_train.loc[train_mask, 'Season']
    train_aug_lgb = augment_data(train_temp_lgb, features, target, noise_scale=lgb_noise)
    weights_aug_lgb = get_recency_sample_weights(train_aug_lgb['Season'], max_season=val_year)
    y_tr_smooth_lgb = label_smoothing(train_aug_lgb[target])
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(train_aug_lgb[features], y_tr_smooth_lgb, sample_weight=weights_aug_lgb)
    p_lgb = model_lgb.predict(X_val)
    
    # --- CatBoost ---
    train_temp_cat = pd.concat([X_tr, y_tr], axis=1)
    train_temp_cat['Season'] = df_train.loc[train_mask, 'Season']
    train_aug_cat = augment_data(train_temp_cat, features, target, noise_scale=cat_noise)
    weights_aug_cat = get_recency_sample_weights(train_aug_cat['Season'], max_season=val_year)
    y_tr_smooth_cat = label_smoothing(train_aug_cat[target])
    
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(train_aug_cat[features], y_tr_smooth_cat, sample_weight=weights_aug_cat)
    p_cat = np.clip(model_cat.predict(X_val), 0.0, 1.0)
    
    temp_df = pd.DataFrame({
        'Season': val_year,
        'XGB_Pred': p_xgb,
        'LGB_Pred': p_lgb,
        'CAT_Pred': p_cat,
        'Target': y_val.values
    })
    OOF_df_list.append(temp_df)
    
    print(f"   [Season {val_year}] XGB: {brier_score_loss(y_val, p_xgb):.4f} | LGB: {brier_score_loss(y_val, p_lgb):.4f} | CAT: {brier_score_loss(y_val, p_cat):.4f}")

oof_df = pd.concat(OOF_df_list, ignore_index=True)

# -------------------------------------------------------------------------
# 7. Level-2 메타 모델 학습 및 캘리브레이션 (Stacking 검증)
# -------------------------------------------------------------------------
print("\n🚀 2단계: Logistic Regression 메타-러너 연도별 타임스플릿 평가 시작...")
final_brier_per_year = []

for val_year in val_seasons:
    # 해당 연도는 Validation Set, 이전 연도들은 Meta-Train Set으로 사용
    meta_train = oof_df[oof_df['Season'] < val_year]
    meta_val = oof_df[oof_df['Season'] == val_year]
    
    if len(meta_train) == 0:
        # 최초 연도(2013)의 경우 메타 훈련데이터가 없으므로 2013-2015 등 초기 데이터의 OOF를 빌려오거나
        # V6 방식의 (3등분 평균)을 임시로 씁니다 (여기서는 간단히 0.33 가중 처리)
        p_stack = (meta_val['XGB_Pred'] + meta_val['LGB_Pred'] + meta_val['CAT_Pred']) / 3.0
    else:
        # 강력한 보정 성능을 가진 페널티 부과 LogisticRegression 탑재
        meta_features = ['XGB_Pred', 'LGB_Pred', 'CAT_Pred']
        X_meta_tr = meta_train[meta_features].values
        y_meta_tr = meta_train['Target'].values
        
        X_meta_val = meta_val[meta_features].values
        y_meta_val = meta_val['Target'].values
        
        # Logistic Regression이 확률 스케일 보정에 최고 (C 파라미터로 정규화)
        meta_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        meta_lr.fit(X_meta_tr, y_meta_tr)
        
        # 확률 (클래스 1) 추출
        p_stack = meta_lr.predict_proba(X_meta_val)[:, 1]
    
    year_brier = brier_score_loss(meta_val['Target'], p_stack)
    final_brier_per_year.append(year_brier)
    print(f"| {val_year} | {year_brier:.4f} |")

print("-" * 60)
print(f"🎯 최종 스태킹(V7) 앙상블 평균 Brier Score: {np.mean(final_brier_per_year):.4f}")
print(f"⏱️ 소요 시간: {(time.time()-START_TIME)/60:.1f}분")
print("-" * 60)
