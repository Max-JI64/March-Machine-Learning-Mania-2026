import os, glob, time, json, random, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

SEED = 42
DATA_DIR = '../Data/provided'
PREP_DIR = '../Data/preprocessed'
NOISE_SCALE = 0.02
# For fast tuning, use fewer seeds and subset of years
TUNE_SEEDS = [42, 123]
TUNE_YEARS = [2023, 2024, 2025]
FULL_SEEDS = [42, 123, 456, 789, 2024]
FULL_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

N_TRIALS = 30 # Optuna trials per model per gender

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_advanced_elo(df_games_sorted, K=20, HOME_ADV=75, REVERSION=0.30):
    elo_dict = {}; elo_records = []
    current_season = df_games_sorted['Season'].iloc[0] if len(df_games_sorted) > 0 else 2000
    for idx, row in df_games_sorted.iterrows():
        season = row['Season']; w_team = row['WTeamID']; l_team = row['LTeamID']; w_loc = row.get('WLoc', 'N'); margin = row['WScore'] - row['LScore']
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
        margin_vector[w] += margin; margin_vector[l] -= margin; games_matrix[w, l] += 1; games_matrix[l, w] += 1; games_played[w] += 1; games_played[l] += 1
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
    
    # Margin
    tourney_results['Margin'] = tourney_results.apply(
        lambda r: r['WScore'] - r['LScore'] if r['T1'] == r['WTeamID'] else r['LScore'] - r['WScore'], axis=1
    )
    
    df = tourney_results[['Season', 'DayNum', 'T1', 'T2', 'WTeamID', 'LTeamID', 'WScore', 'LScore', 'Label', 'Margin']]
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

def augment_data_margin(df, features, target='Margin', noise_scale=0.02):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    for pc in ['EloWinProb', 'SeedWinProb']:
        if pc in features: df_swap[pc] = 1 - df_swap[pc]
    for f in ['Momentum_Upset_Risk', 'Tournament_DNA_Risk']:
        if f in features: df_swap[f] = -df_swap[f]
    
    df_swap[target] = -df_swap[target] 
    
    df_aug = pd.concat([df, df_swap], ignore_index=True)
    numeric_feats = [f for f in features if f in df_aug.columns]
    feat_stds = df_aug[numeric_feats].std().values; feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(len(df_aug), len(numeric_feats)))
    df_aug[numeric_feats] += (noise * feat_stds)
    return df_aug

def convert_margin_to_prob(margin, std_dev):
    return norm.cdf(margin / std_dev)

# Pre-cache training data to speed up Optuna
def prep_optuna_cache(df, features, years, seeds):
    cache = []
    for val_year in years:
        train_mask = df['Season'] != val_year
        val_mask = df['Season'] == val_year
        X_va = df.loc[val_mask, features].values
        y_va_margin = df.loc[val_mask, 'Margin'].values
        
        for seed in seeds:
            set_seed(seed)
            df_tr_aug = augment_data_margin(df[train_mask], features, 'Margin', noise_scale=NOISE_SCALE)
            X_tr = df_tr_aug[features].values
            y_tr = df_tr_aug['Margin'].values
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            
            cache.append({
                'seed': seed, 'val_year': val_year,
                'X_tr': X_tr, 'y_tr': y_tr,
                'X_va': X_va, 'y_va_margin': y_va_margin,
                'X_tr_sc': X_tr_sc, 'X_va_sc': X_va_sc
            })
    return cache

# Objectives for Margin (MSE optimization)
def run_model_trial(cache, model_class, params, use_scaled=False):
    mse_list = []
    for c in cache:
        seed = c['seed']
        params_w_seed = params.copy()
        if model_class == Ridge:
            params_w_seed['random_state'] = seed
        elif model_class == CatBoostRegressor:
            params_w_seed['random_seed'] = seed; params_w_seed['verbose'] = False
        elif model_class == XGBRegressor:
            params_w_seed['random_state'] = seed; params_w_seed['verbosity'] = 0
        elif model_class == LGBMRegressor:
            params_w_seed['random_state'] = seed; params_w_seed['verbose'] = -1; params_w_seed['n_jobs'] = -1
            
        m = model_class(**params_w_seed)
        
        X_t = c['X_tr_sc'] if use_scaled else c['X_tr']
        X_v = c['X_va_sc'] if use_scaled else c['X_va']
        
        m.fit(X_t, c['y_tr'])
        preds = m.predict(X_v)
        mse_list.append(mean_squared_error(c['y_va_margin'], preds))
    return np.mean(mse_list)

def optimize_models(cache, gender):
    print(f"\n--- Optimizing for {gender} (MSE Target) ---")
    
    # 1. Ridge
    def obj_ridge(trial):
        alpha = trial.suggest_float('alpha', 0.1, 200.0, log=True)
        return run_model_trial(cache, Ridge, {'alpha': alpha}, use_scaled=True)
    
    study_ridge = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_ridge.optimize(obj_ridge, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"[{gender}] Ridge Best MSE: {study_ridge.best_value:.3f} | params: {study_ridge.best_params}")
    
    # 2. XGBoost
    def obj_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        }
        return run_model_trial(cache, XGBRegressor, params)

    study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(obj_xgb, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"[{gender}] XGB Best MSE: {study_xgb.best_value:.3f} | params: {study_xgb.best_params}")

    # 3. LightGBM
    def obj_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'num_leaves': trial.suggest_int('num_leaves', 10, 63),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'objective': 'regression'
        }
        return run_model_trial(cache, LGBMRegressor, params)

    study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgb.optimize(obj_lgb, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"[{gender}] LGB Best MSE: {study_lgb.best_value:.3f} | params: {study_lgb.best_params}")
    
    # 4. CatBoost
    def obj_cat(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 6),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 50.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 3.0),
            'border_count': trial.suggest_int('border_count', 64, 254),
        }
        return run_model_trial(cache, CatBoostRegressor, params)

    study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_cat.optimize(obj_cat, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"[{gender}] CatBoost Best MSE: {study_cat.best_value:.3f} | params: {study_cat.best_params}")

    return {
        'ridge': study_ridge.best_params,
        'xgb': study_xgb.best_params,
        'lgb': study_lgb.best_params,
        'cat': study_cat.best_params
    }


if __name__ == "__main__":
    START_TIME = time.time()
    set_seed(SEED)

    print("=" * 80)
    print("V62: Margin-Specific Hyperparameter Deep Tuning")
    print(f"Trials per model: {N_TRIALS} | Tuning on {TUNE_YEARS}")
    print("=" * 80)

    print("\nLoading data...", flush=True)
    df_m = load_data('M')
    df_w = load_data('W')

    with open('selected_features_v7.json') as f:
        v7_features = json.load(f)
    full_req = list(set(v7_features + ['EloWinProb', 'SeedWinProb', 'Elo_x_SRS_Diff',
                   'WinRate_x_Pyth_Diff', 'MasMean_Diff', 'Momentum_Upset_Risk',
                   'Tournament_DNA_Risk', 'AstTO_Ratio_Diff', 'OffRtg_Diff']))
    features_m = [f for f in full_req if f in df_m.columns and f not in
                  ['Is_Major_Conf_Diff', 'Elo_Diff', 'LateSeason_Altitude_Fatigue_Diff', '3PAr_Diff', 'OR%_Diff']]
    features_w = [f for f in full_req if f in df_w.columns and f not in
                  ['MasMean_Diff', 'WinRate_vs_Top50_Diff', 'Upset_Value_Diff', 'Coach_Tenure_Diff',
                   'Rank_Trend_Slope_Diff', 'Is_Major_Conf_Diff']]

    print(f"Men features: {len(features_m)} | Women features: {len(features_w)}", flush=True)

    print("\nPre-computing data cache for fast Optuna evaluation...")
    cache_m = prep_optuna_cache(df_m, features_m, TUNE_YEARS, TUNE_SEEDS)
    cache_w = prep_optuna_cache(df_w, features_w, TUNE_YEARS, TUNE_SEEDS)
    
    print("\nPhase 1: Tuning Hyperparameters (MSE Objective)")
    best_params_m = optimize_models(cache_m, 'M')
    best_params_w = optimize_models(cache_w, 'W')
    
    print("\nPhase 2: Generating Full 13-Year Predictions with New Params")
    # For speed, we will output the params to a json and run the full evaluation in V63.
    # V62 focuses on finding the best MSE params.
    
    config = {
        'best_params_m': best_params_m,
        'best_params_w': best_params_w
    }
    with open('v62_margin_params.json', 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"\nAll tuning finished in {(time.time() - START_TIME)/60:.1f} minutes.")
    print("Parameters saved to v62_margin_params.json")
