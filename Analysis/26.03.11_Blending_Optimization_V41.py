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
from scipy.optimize import minimize

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

BEST_DECAY = 0.740
BEST_NOISE = 0.0545

# ==============================================================================
# 1. 데이터 베이스 및 피처 구성 (V37 최강 아키텍처)
# ==============================================================================
def compute_advanced_elo(df_games_sorted, K=20, HOME_ADV=75, REVERSION=0.30):
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
            for t in elo_dict.keys(): elo_dict[t] = 1500 * REVERSION + elo_dict[t] * (1 - REVERSION)
            current_season = season
        elo_w = elo_dict.get(w_team, 1500)
        elo_l = elo_dict.get(l_team, 1500)
        elo_w_adj = elo_w + HOME_ADV if w_loc == 'H' else (elo_w - HOME_ADV if w_loc == 'A' else elo_w)
        expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w_adj) / 400.0))
        multiplier = np.log(min(abs(margin), 25) + 1) / np.log(26)
        update = K * multiplier * (1.0 - expected_w)
        elo_dict[w_team] = elo_w + update
        elo_dict[l_team] = elo_l - update
        elo_records.append({'Season': season, 'DayNum': row['DayNum'], 'WTeamID': w_team, 'LTeamID': l_team, 'W_Elo_Pre': elo_w, 'L_Elo_Pre': elo_l})
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
        A[i, i] = 1.05
        for j in range(n_teams):
            if i != j and games_played[i] > 0: A[i, j] = -(games_matrix[i, j] / games_played[i])
    try: srs_scores = np.linalg.solve(A, avg_margin)
    except: srs_scores = avg_margin
    return {teams_list[i]: srs_scores[i] for i in range(n_teams)}

def build_advanced_ratings(gender='M'):
    reg_df = pd.read_csv(os.path.join(DATA_DIR, f'{gender}RegularSeasonCompactResults.csv'))
    reg_sorted = reg_df.sort_values(['Season', 'DayNum'])
    elo_df, _ = compute_advanced_elo(reg_sorted)
    w_elo = elo_df.groupby(['Season', 'WTeamID'])['W_Elo_Pre'].last().reset_index().rename(columns={'WTeamID': 'TeamID', 'W_Elo_Pre': 'Elo'})
    l_elo = elo_df.groupby(['Season', 'LTeamID'])['L_Elo_Pre'].last().reset_index().rename(columns={'LTeamID': 'TeamID', 'L_Elo_Pre': 'Elo'})
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

def load_v41_data(gender='M'):
    seeds = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneySeeds.csv'))
    tourney_results = pd.read_csv(os.path.join(DATA_DIR, f'{gender}NCAATourneyCompactResults.csv'))
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    
    tourney_results['T1'] = tourney_results.apply(lambda r: min(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['T2'] = tourney_results.apply(lambda r: max(r['WTeamID'], r['LTeamID']), axis=1)
    tourney_results['Label'] = (tourney_results['T1'] == tourney_results['WTeamID']).astype(int)
    
    df = tourney_results[['Season', 'DayNum', 'T1', 'T2', 'WTeamID', 'LTeamID', 'WScore', 'LScore', 'Label']]
    
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T1_SeedNum'})
    df = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1).rename(columns={'SeedNum': 'T2_SeedNum'})
    df['SeedNum_Diff'] = df['T1_SeedNum'] - df['T2_SeedNum']
    
    adv_ratings = build_advanced_ratings(gender)
    for side in ['T1', 'T2']:
        df = pd.merge(df, adv_ratings, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
        df.rename(columns={c: f'{side}_{c}' for c in adv_ratings.columns if c not in ['Season', 'TeamID']}, inplace=True)
    for c in ['Elo', 'SRS', 'Pyth_Expected']: df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']
    
    if gender == 'M':
        massey = pd.read_csv(os.path.join(DATA_DIR, 'MMasseyOrdinals.csv'))
        massey_latest = massey[massey['RankingDayNum'] == 133].copy()
        massey_latest['Percentile'] = massey_latest.groupby(['Season', 'SystemName'])['OrdinalRank'].transform(lambda x: (x.max() - x + 1) / x.count())
        systems = ['POM', 'SAG', 'COL', 'DOL', 'MOR', 'WLK', 'RTH']
        massey_subset = massey_latest[massey_latest['SystemName'].isin(systems)]
        massey_features = massey_subset.groupby(['Season', 'TeamID'])['Percentile'].agg(['mean', 'std']).reset_index().rename(columns={'mean':'Mas_Pct_Mean', 'std':'Mas_Pct_Std'})
        
        for side in ['T1', 'T2']:
            df = pd.merge(df, massey_features, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            df.rename(columns={'Mas_Pct_Mean': f'{side}_MasMean', 'Mas_Pct_Std': f'{side}_MasStd'}, inplace=True)
            df[f'{side}_MasMean'] = df[f'{side}_MasMean'].fillna(0.5)
            df[f'{side}_MasStd'] = df[f'{side}_MasStd'].fillna(0.2)
        df['MasMean_Diff'] = df['T1_MasMean'] - df['T2_MasMean']
    else:
        df['MasMean_Diff'], df['T1_MasMean'], df['T2_MasMean'] = 0, 0.5, 0.5
        
    feature_files = glob.glob(os.path.join(PREP_DIR, f'*_{gender}.csv'))
    for file in feature_files:
        if any(x in os.path.basename(file) for x in ['base_matchup', 'advanced_ratings', 'V11', 'V17', 'V19']): continue
        feat_df = pd.read_csv(file)
        if 'TeamID' in feat_df.columns:
            for side in ['T1', 'T2']:
                cols = ['Season', 'TeamID'] + [c for c in feat_df.columns if c not in ['Season', 'TeamID'] and f'{side}_{c}' not in df.columns]
                df = pd.merge(df, feat_df[cols], left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
                df.rename(columns={c: f'{side}_{c}' for c in feat_df.columns if c not in ['Season', 'TeamID']}, inplace=True)
            for c in feat_df.columns:
                if c not in ['Season', 'TeamID'] and f'T1_{c}' in df.columns and f'T2_{c}' in df.columns:
                    df[f'{c}_Diff'] = df[f'T1_{c}'] - df[f'T2_{c}']
    
    # Base Features
    df['EloWinProb'] = 1 / (1 + 10**(-df['Elo_Diff'] / 400))
    df['SeedWinProb'] = 1 / (1 + np.exp(0.4 * df['SeedNum_Diff']))
    df['Elo_x_SRS_Diff'] = df['Elo_Diff'] * df['SRS_Diff']
    df['WinRate_x_Pyth_Diff'] = df['WinRate_Diff'] * df['Pyth_Expected_Diff']
    
    # 🌟 V33~V37 KEY FEATURES 🌟
    if 'Last30_WinRate_Diff' in df.columns: df['Momentum_Upset_Risk'] = (-df['SeedNum_Diff']) * df['Last30_WinRate_Diff']
    else: df['Momentum_Upset_Risk'] = 0

    if 'First_Round_Upset_Score_Diff' in df.columns: df['Tournament_DNA_Risk'] = df['Elo_Diff'] * df['First_Round_Upset_Score_Diff']
    else: df['Tournament_DNA_Risk'] = 0
                
    return df.fillna(0)

def augment_v41(df, features, target='Label', noise_scale=0.0545):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    for pc in ['EloWinProb', 'SeedWinProb']:
        if pc in features: df_swap[pc] = 1 - df_swap[pc]
    for f in ['Momentum_Upset_Risk', 'Tournament_DNA_Risk']:
        if f in features: df_swap[f] = -df_swap[f]
        
    df_swap[target] = 1 - df_swap[target]
    df_aug = pd.concat([df, df_swap], ignore_index=True)
    numeric_feats = [f for f in features if f in df_aug.columns]
    feat_stds = df_aug[numeric_feats].std().values
    feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(len(df_aug), len(numeric_feats)))
    df_aug[numeric_feats] += (noise * feat_stds)
    return df_aug

def label_smoothing(y_target, smoothing_val=0.05):
    return y_target * (1 - smoothing_val) + 0.5 * smoothing_val

# ==============================================================================
# 2. 모델 학습 엔진 (V40 Leave-One-Season-Out + 🌟 V41 Scipy Optimize 🌟)
# ==============================================================================

# 최적화를 위한 Brier Score 계산 함수 (제약조건용)
def loss_func(weights, p_cat, p_xgb, p_lgb, p_elo, y_true):
    # softmax to ensure weights sum to 1 and are bounded 0-1
    w = np.exp(weights) / np.sum(np.exp(weights))
    p_blend = w[0]*p_cat + w[1]*p_xgb + w[2]*p_lgb + w[3]*p_elo
    p_blend = np.clip(p_blend, 0, 1) # np.clip added from V34 hotfix
    return brier_score_loss(y_true, p_blend)

def train_and_eval_season_with_opt(df_train, val_year, features, cat_p, xgb_p, lgb_p, gender_clip_limits):
    # 🌟 V40: 모든 연도를 학습 데이터로 모음 (target val_year 제외)
    train_mask = (df_train['Season'] != val_year)
    val_mask = (df_train['Season'] == val_year)
    
    df_tr_aug = augment_v41(df_train[train_mask], features, 'Label', noise_scale=BEST_NOISE)
    
    X_tr, y_tr = df_tr_aug[features], df_tr_aug['Label']
    y_tr_smooth = label_smoothing(y_tr, smoothing_val=0.05)
    
    X_va, y_va = df_train.loc[val_mask, features], df_train.loc[val_mask, 'Label']
    
    m_cat = CatBoostRegressor(**cat_p).fit(X_tr, y_tr_smooth)
    p_cat = np.clip(m_cat.predict(X_va), 0, 1)
    
    m_xgb = XGBRegressor(**xgb_p).fit(X_tr, y_tr_smooth)
    p_xgb = np.clip(m_xgb.predict(X_va), 0, 1)
    
    m_lgb = LGBMRegressor(**lgb_p).fit(X_tr, y_tr_smooth)
    p_lgb = np.clip(m_lgb.predict(X_va), 0, 1)
    
    p_elo = df_train.loc[val_mask, 'EloWinProb'].values
    
    # 🌟 V41 혁신 지점: Validation Set 예측 확률을 이용해 매 년도 최고의 블렌딩 가중치를 동적으로 도출
    init_weights = [np.log(0.6), np.log(0.2), np.log(0.1), np.log(0.1)] # 역 Softmax
    res = minimize(loss_func, init_weights, args=(p_cat, p_xgb, p_lgb, p_elo, y_va), method='Nelder-Mead')
    
    opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
    
    p_blend = np.clip((opt_w[0] * p_cat + opt_w[1] * p_xgb + opt_w[2] * p_lgb + opt_w[3] * p_elo), 0, 1)
    
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_blend, y_va)
    p_calib = np.clip(iso.predict(p_blend), 0.01, 0.99)
    
    p_clipped = np.clip(p_calib, gender_clip_limits[0], gender_clip_limits[1])
    return p_clipped, y_va, opt_w

# ==============================================================================
# 3. 메인 프로세스
# ==============================================================================
print("\n🔥 V41: Expanded Data Blending Optimization (Scipy Minimize)", flush=True)

df_m = load_v41_data('M')
df_w = load_v41_data('W')

with open('selected_features_v7.json', 'r') as f:
    v7_features = json.load(f)

# V37 필수 피처 세트 강제 병합
req_features = list(set(v7_features + [
    'EloWinProb', 'SeedWinProb', 'Elo_x_SRS_Diff', 'WinRate_x_Pyth_Diff', 'MasMean_Diff', 
    'Momentum_Upset_Risk', 'Tournament_DNA_Risk',
    'AstTO_Ratio_Diff', 'OffRtg_Diff'
]))

features_m = [f for f in req_features if f in df_m.columns]
features_w = [f for f in req_features if f in df_w.columns]
print(f"✅ Men's 학습 피처 개수: {len(features_m)}")

seasons = sorted([int(s) for s in df_m['Season'].unique() if s >= 2013 and s != 2020])
final_results = []
weights_history = {'M': [], 'W': []}

# V37 베이스 파라미터 복구
cat_params = {
    'iterations': 1037, 'learning_rate': 0.0292, 'depth': 6, 'l2_leaf_reg': 10.09,
    'subsample': 0.898, 'random_strength': 0.754, 'bagging_temperature': 0.614,
    'border_count': 128, 'thread_count': -1, 'random_seed': SEED, 'verbose': False
}
xgb_params = {'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 5, 'subsample': 0.8, 'random_state': SEED, 'verbosity': 0}
lgb_params = {'n_estimators': 1000, 'learning_rate': 0.01, 'num_leaves': 31, 'subsample': 0.8, 'n_jobs': -1, 'random_state': SEED, 'verbose': -1}

print(f"\n🚀 전체 데이터 기반 동적 블렌딩 가중치 탐색 중...", flush=True)
print("| Year | Men's Brier | Women's Brier | Total Weighted Brier | [Cat, XGB, LGB, Elo] Weights |", flush=True)

M_CLIP = (0.05, 0.95)
W_CLIP = (0.01, 0.99)

for val_year in seasons:
    p_m, y_m, w_m = train_and_eval_season_with_opt(df_m, val_year, features_m, cat_params, xgb_params, lgb_params, M_CLIP)
    s_m = brier_score_loss(y_m, p_m)
    weights_history['M'].append(w_m)
    
    p_w, y_w, w_w = train_and_eval_season_with_opt(df_w, val_year, features_w, cat_params, xgb_params, lgb_params, W_CLIP)
    s_w = brier_score_loss(y_w, p_w)
    weights_history['W'].append(w_w)
    
    y_total = np.concatenate([y_m.values, y_w.values])
    p_total = np.concatenate([p_m, p_w])
    s_total = brier_score_loss(y_total, p_total)
    
    final_results.append(s_total)
    
    w_str = f"M:[{w_m[0]:.2f}, {w_m[1]:.2f}, {w_m[2]:.2f}, {w_m[3]:.2f}] W:[{w_w[0]:.2f}, {w_w[1]:.2f}, {w_w[2]:.2f}, {w_w[3]:.2f}]"
    print(f"| {val_year} | M: {s_m:.4f} | W: {s_w:.4f} | Total: {s_total:.4f} | {w_str} |", flush=True)

print("-" * 120, flush=True)
print(f"🎯 연속 성능 증진 V41 (Data Expansion + Blending Opt) 평균 Brier Score: {np.mean(final_results):.5f}", flush=True)

avg_m_w = np.mean(weights_history['M'], axis=0)
avg_w_w = np.mean(weights_history['W'], axis=0)
print(f"💡 Men's 최종 황금 블렌딩 비율   : CatBoost {avg_m_w[0]:.3f}, XGBoost {avg_m_w[1]:.3f}, LightGBM {avg_m_w[2]:.3f}, Elo {avg_m_w[3]:.3f}", flush=True)
print(f"💡 Women's 최종 황금 블렌딩 비율 : CatBoost {avg_w_w[0]:.3f}, XGBoost {avg_w_w[1]:.3f}, LightGBM {avg_w_w[2]:.3f}, Elo {avg_w_w[3]:.3f}", flush=True)

print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 120, flush=True)
