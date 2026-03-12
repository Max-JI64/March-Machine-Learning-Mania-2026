import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
import warnings
from scipy.optimize import minimize
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

# ==============================================================================
# 1. 데이터 베이스 및 피처 구성 (V43 베이스)
# ==============================================================================
def compute_advanced_elo(df_games_sorted, K=20, HOME_ADV=75, REVERSION=0.30):
    elo_dict = {}
    elo_records = []
    current_season = df_games_sorted['Season'].iloc[0] if len(df_games_sorted) > 0 else 2000
    for idx, row in df_games_sorted.iterrows():
        season = row['Season']
        w_team = row['WTeamID']; l_team = row['LTeamID']
        w_loc = row.get('WLoc', 'N')
        margin = row['WScore'] - row['LScore']
        if season != current_season:
            for t in elo_dict.keys(): elo_dict[t] = 1500 * REVERSION + elo_dict[t] * (1 - REVERSION)
            current_season = season
        elo_w = elo_dict.get(w_team, 1500); elo_l = elo_dict.get(l_team, 1500)
        elo_w_adj = elo_w + HOME_ADV if w_loc == 'H' else (elo_w - HOME_ADV if w_loc == 'A' else elo_w)
        expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w_adj) / 400.0))
        multiplier = np.log(min(abs(margin), 25) + 1) / np.log(26)
        update = K * multiplier * (1.0 - expected_w)
        elo_dict[w_team] = elo_w + update; elo_dict[l_team] = elo_l - update
        elo_records.append({'Season': season, 'DayNum': row['DayNum'], 'WTeamID': w_team, 'LTeamID': l_team, 'W_Elo_Pre': elo_w, 'L_Elo_Pre': elo_l})
    return pd.DataFrame(elo_records), elo_dict

def compute_srs(df_season_games, teams_list):
    n_teams = len(teams_list); t_idx = {t: i for i, t in enumerate(teams_list)}
    margin_vector = np.zeros(n_teams); games_matrix = np.zeros((n_teams, n_teams)); games_played = np.zeros(n_teams)
    for _, row in df_season_games.iterrows():
        if row['WTeamID'] not in t_idx or row['LTeamID'] not in t_idx: continue
        w, l = t_idx[row['WTeamID']], t_idx[row['LTeamID']]
        margin = row['WScore'] - row['LScore']
        margin_vector[w] += margin; margin_vector[l] -= margin
        games_matrix[w, l] += 1; games_matrix[l, w] += 1
        games_played[w] += 1; games_played[l] += 1
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
        s_df = reg_df[reg_df['Season'] == s]; teams = set(s_df['WTeamID']).union(set(s_df['LTeamID']))
        srs_dict = compute_srs(s_df, list(teams))
        for t, val in srs_dict.items(): srs_records.append({'Season': s, 'TeamID': t, 'SRS': val})
    team_srs = pd.DataFrame(srs_records)
    w_stats = reg_df.groupby(['Season', 'WTeamID']).agg(WScore=('WScore', 'mean'), LScore=('LScore', 'mean')).reset_index().rename(columns={'WTeamID':'TeamID'})
    l_stats = reg_df.groupby(['Season', 'LTeamID']).agg(LScore=('LScore', 'mean'), WScore=('WScore', 'mean')).reset_index().rename(columns={'LTeamID':'TeamID'})
    cmb = pd.concat([w_stats, l_stats]).groupby(['Season', 'TeamID']).mean().reset_index()
    cmb['Pyth_Expected'] = (cmb['WScore']**11.5) / ((cmb['WScore']**11.5) + (cmb['LScore']**11.5) + 1e-9)
    return team_elodf.merge(team_srs, on=['Season', 'TeamID'], how='left').merge(cmb[['Season', 'TeamID', 'Pyth_Expected']], on=['Season', 'TeamID'], how='left')

def load_data(gender='M'):
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
        massey = pd.read_csv(os.path.join(DATA_DIR, 'MMasseyOrdinals.csv')); massey_latest = massey[massey['RankingDayNum'] == 133].copy()
        massey_latest['Percentile'] = massey_latest.groupby(['Season', 'SystemName'])['OrdinalRank'].transform(lambda x: (x.max() - x + 1) / x.count())
        systems = ['POM', 'SAG', 'COL', 'DOL', 'MOR', 'WLK', 'RTH']; massey_subset = massey_latest[massey_latest['SystemName'].isin(systems)]
        massey_features = massey_subset.groupby(['Season', 'TeamID'])['Percentile'].agg(['mean', 'std']).reset_index().rename(columns={'mean':'Mas_Pct_Mean', 'std':'Mas_Pct_Std'})
        for side in ['T1', 'T2']:
            df = pd.merge(df, massey_features, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            df.rename(columns={'Mas_Pct_Mean': f'{side}_MasMean', 'Mas_Pct_Std': f'{side}_MasStd'}, inplace=True)
            df[f'{side}_MasMean'] = df[f'{side}_MasMean'].fillna(0.5); df[f'{side}_MasStd'] = df[f'{side}_MasStd'].fillna(0.2)
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
    df['EloWinProb'] = 1 / (1 + 10**(-df['Elo_Diff'] / 400)); df['SeedWinProb'] = 1 / (1 + np.exp(0.4 * df['SeedNum_Diff']))
    df['Elo_x_SRS_Diff'] = df['Elo_Diff'] * df['SRS_Diff']; df['WinRate_x_Pyth_Diff'] = df['WinRate_Diff'] * df['Pyth_Expected_Diff']
    if 'Last30_WinRate_Diff' in df.columns: df['Momentum_Upset_Risk'] = (-df['SeedNum_Diff']) * df['Last30_WinRate_Diff']
    else: df['Momentum_Upset_Risk'] = 0
    if 'First_Round_Upset_Score_Diff' in df.columns: df['Tournament_DNA_Risk'] = df['Elo_Diff'] * df['First_Round_Upset_Score_Diff']
    else: df['Tournament_DNA_Risk'] = 0
    return df.fillna(0)

def augment_data(df, features, target='Label', noise_scale=0.0545):
    df_swap = df.copy(); diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    for pc in ['EloWinProb', 'SeedWinProb']:
        if pc in features: df_swap[pc] = 1 - df_swap[pc]
    for f in ['Momentum_Upset_Risk', 'Tournament_DNA_Risk']:
        if f in features: df_swap[f] = -df_swap[f]
    df_swap[target] = 1 - df_swap[target]
    df_aug = pd.concat([df, df_swap], ignore_index=True)
    numeric_feats = [f for f in features if f in df_aug.columns]
    feat_stds = df_aug[numeric_feats].std().values; feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(len(df_aug), len(numeric_feats)))
    df_aug[numeric_feats] += (noise * feat_stds)
    return df_aug

def label_smoothing(y_target, smoothing_val=0.05):
    return y_target * (1 - smoothing_val) + 0.5 * smoothing_val

# ==============================================================================
# 2. 모델 예측 수집 (OOF)
# ==============================================================================
def get_oof_predictions(df, features, cat_p, xgb_p, lgb_p, val_years):
    oof_preds = []
    
    for vy in val_years:
        if vy not in df['Season'].values:
            continue
            
        print(f"  [Year {vy}] Data prep...", end=' ', flush=True)
        train_mask = (df['Season'] != vy); val_mask = (df['Season'] == vy)
        df_tr_aug = augment_data(df[train_mask], features, 'Label', noise_scale=0.0545)
        X_tr, y_tr = df_tr_aug[features], df_tr_aug['Label']
        y_tr_smooth = label_smoothing(y_tr, smoothing_val=0.05)
        X_va, y_va = df.loc[val_mask, features], df.loc[val_mask, 'Label']
        
        print("CatBoost...", end=' ', flush=True)
        m_cat = CatBoostRegressor(**cat_p, verbose=False, random_seed=SEED).fit(X_tr, y_tr_smooth)
        p_cat = np.clip(m_cat.predict(X_va), 1e-6, 1-1e-6)
        
        print("XGBoost...", end=' ', flush=True)
        m_xgb = XGBRegressor(**xgb_p, random_state=SEED, verbosity=0).fit(X_tr, y_tr_smooth)
        p_xgb = np.clip(m_xgb.predict(X_va), 1e-6, 1-1e-6)
        
        print("LightGBM...", end=' ', flush=True)
        m_lgb = LGBMRegressor(**lgb_p, random_state=SEED, verbose=-1, n_jobs=-1).fit(X_tr, y_tr_smooth)
        p_lgb = np.clip(m_lgb.predict(X_va), 1e-6, 1-1e-6)
        
        p_elo = df.loc[val_mask, 'EloWinProb'].values
        
        oof_preds.append({
            'Season': vy,
            'y_true': y_va.values,
            'p_cat': p_cat,
            'p_xgb': p_xgb,
            'p_lgb': p_lgb,
            'p_elo': p_elo
        })
        print("Done! ✅", flush=True)
        
    return oof_preds

# ==============================================================================
# 3. 가중치 최적화 및 캘리브레이션
# ==============================================================================
def objective_weights(weights, p_cat, p_xgb, p_lgb, p_elo, y_true):
    w_cat, w_xgb, w_lgb, w_elo = weights
    p_blend = w_cat * p_cat + w_xgb * p_xgb + w_lgb * p_lgb + w_elo * p_elo
    p_blend = np.clip(p_blend, 1e-6, 1-1e-6)
    return brier_score_loss(y_true, p_blend)

def optimize_global_weights(oof_data):
    all_y = np.concatenate([d['y_true'] for d in oof_data])
    all_cat = np.concatenate([d['p_cat'] for d in oof_data])
    all_xgb = np.concatenate([d['p_xgb'] for d in oof_data])
    all_lgb = np.concatenate([d['p_lgb'] for d in oof_data])
    all_elo = np.concatenate([d['p_elo'] for d in oof_data])
    
    init_weights = [0.25, 0.25, 0.25, 0.25]
    bounds = [(0, 1)] * 4
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    res = minimize(objective_weights, init_weights, args=(all_cat, all_xgb, all_lgb, all_elo, all_y),
                   method='SLSQP', bounds=bounds, constraints=constraints)
    
    return res.x

# ==============================================================================
# 4. 메인 프로세스
# ==============================================================================
if __name__ == "__main__":
    print("🔥 V48: Final Ensemble & Calibration (Deep Optuna Params + Global Weight Optimization + Isotonic)", flush=True)
    
    df_m = load_data('M')
    df_w = load_data('W')
    
    with open('selected_features_v7.json', 'r') as f: v7_features = json.load(f)
    full_req_features = list(set(v7_features + [
        'EloWinProb', 'SeedWinProb', 'Elo_x_SRS_Diff', 'WinRate_x_Pyth_Diff', 'MasMean_Diff', 
        'Momentum_Upset_Risk', 'Tournament_DNA_Risk', 'AstTO_Ratio_Diff', 'OffRtg_Diff'
    ]))
    
    features_m = [f for f in full_req_features if f in df_m.columns and f not in ['Is_Major_Conf_Diff', 'Elo_Diff', 'LateSeason_Altitude_Fatigue_Diff', '3PAr_Diff', 'OR%_Diff']]
    features_w = [f for f in full_req_features if f in df_w.columns and f not in ['MasMean_Diff', 'WinRate_vs_Top50_Diff', 'Upset_Value_Diff', 'Coach_Tenure_Diff', 'Rank_Trend_Slope_Diff', 'Is_Major_Conf_Diff']]
    
    seasons = sorted([int(s) for s in df_m['Season'].unique() if s >= 2012 and s != 2020])
    
    # Load V47 Best Parameters
    with open('v47_deep_optuna_results.json', 'r') as f:
        v47_params = json.load(f)
        
    m_cat_p = v47_params['Men_CatBoost']
    m_xgb_p = v47_params['Men_XGBoost']
    m_lgb_p = v47_params['Men_LightGBM']
    
    w_cat_p = v47_params['Women_CatBoost']
    w_xgb_p = v47_params['Women_XGBoost']
    w_lgb_p = v47_params['Women_LightGBM']
    
    print("\n🚀 1. Collecting OOF Predictions for Men...")
    oof_m = get_oof_predictions(df_m, features_m, m_cat_p, m_xgb_p, m_lgb_p, seasons)
    
    print("\n🚀 2. Collecting OOF Predictions for Women...")
    oof_w = get_oof_predictions(df_w, features_w, w_cat_p, w_xgb_p, w_lgb_p, seasons)
    
    print("\n🚀 3. Optimizing Global Blending Weights...")
    best_w_m = optimize_global_weights(oof_m)
    best_w_w = optimize_global_weights(oof_w)
    
    print(f"💡 Men's Global Best Weights   : CatBoost {best_w_m[0]:.3f}, XGBoost {best_w_m[1]:.3f}, LightGBM {best_w_m[2]:.3f}, Elo {best_w_m[3]:.3f}")
    print(f"💡 Women's Global Best Weights : CatBoost {best_w_w[0]:.3f}, XGBoost {best_w_w[1]:.3f}, LightGBM {best_w_w[2]:.3f}, Elo {best_w_w[3]:.3f}")
    
    print("\n🚀 4. Applying Isotonic Calibration & Evaluating...")
    
    # Global Isotonic Calibration
    # Men
    all_y_m = np.concatenate([d['y_true'] for d in oof_m])
    all_blend_m = np.concatenate([best_w_m[0]*d['p_cat'] + best_w_m[1]*d['p_xgb'] + best_w_m[2]*d['p_lgb'] + best_w_m[3]*d['p_elo'] for d in oof_m])
    iso_m = IsotonicRegression(out_of_bounds='clip').fit(all_blend_m, all_y_m)
    
    # Women
    all_y_w = np.concatenate([d['y_true'] for d in oof_w])
    all_blend_w = np.concatenate([best_w_w[0]*d['p_cat'] + best_w_w[1]*d['p_xgb'] + best_w_w[2]*d['p_lgb'] + best_w_w[3]*d['p_elo'] for d in oof_w])
    iso_w = IsotonicRegression(out_of_bounds='clip').fit(all_blend_w, all_y_w)
    
    final_results = []
    print("\n| Year | Men's Brier | Women's Brier | Total Weighted Brier |")
    
    for val_year in seasons:
        # Men
        d_m = next((d for d in oof_m if d['Season'] == val_year), None)
        if d_m:
            p_blend_m = best_w_m[0]*d_m['p_cat'] + best_w_m[1]*d_m['p_xgb'] + best_w_m[2]*d_m['p_lgb'] + best_w_m[3]*d_m['p_elo']
            p_calib_m = np.clip(iso_m.predict(p_blend_m), 0.05, 0.95)
            s_m = brier_score_loss(d_m['y_true'], p_calib_m)
            y_total = d_m['y_true']
            p_total = p_calib_m
        else:
            s_m = 0.0; y_total = np.array([]); p_total = np.array([])
            
        # Women
        d_w = next((d for d in oof_w if d['Season'] == val_year), None)
        if d_w:
            p_blend_w = best_w_w[0]*d_w['p_cat'] + best_w_w[1]*d_w['p_xgb'] + best_w_w[2]*d_w['p_lgb'] + best_w_w[3]*d_w['p_elo']
            p_calib_w = np.clip(iso_w.predict(p_blend_w), 0.01, 0.99)
            s_w = brier_score_loss(d_w['y_true'], p_calib_w)
            y_total = np.concatenate([y_total, d_w['y_true']])
            p_total = np.concatenate([p_total, p_calib_w])
        else:
            s_w = 0.0
            
        s_total = brier_score_loss(y_total, p_total) if len(y_total) > 0 else 0.0
        final_results.append(s_total)
        print(f"| {val_year} | M: {s_m:.4f} | W: {s_w:.4f} | Total: {s_total:.4f} |", flush=True)

    print("-" * 100, flush=True)
    print(f"🎯 최후의 0.12 고지 함락 V48 (Deep Params + Global Weights + Isotonic) 평균 Brier Score: {np.mean(final_results):.5f}", flush=True)
    print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
    print("-" * 100, flush=True)
