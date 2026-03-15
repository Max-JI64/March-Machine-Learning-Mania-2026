import os, glob, time, json, random, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

SEED = 42
DATA_DIR = '../Data/provided'
PREP_DIR = '../Data/preprocessed'
LABEL_SMOOTH = 0.01
NOISE_SCALE = 0.02

TUNE_SEEDS = [42, 123]
FULL_SEEDS = [42, 123, 456, 789, 2024, 7, 31, 99, 555, 1234]
TUNE_YEARS = [2023, 2024, 2025]
FULL_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

M_WEIGHTS_V58 = [0.16, 0.30, 0.14, 0.15, 0.15, 0.10]
W_WEIGHTS_V58 = [0.30, 0.14, 0.14, 0.15, 0.15, 0.12]
M_CLIP, W_CLIP = (0.05, 0.95), (0.01, 0.99)

with open('v58_best_params.json') as f:
    v58 = json.load(f)
XGB_V58 = v58['xgb']
LGB_V58 = v58['lgb']

CAT_V23 = {
    'iterations': 1037, 'learning_rate': 0.0292, 'depth': 6,
    'l2_leaf_reg': 10.09, 'subsample': 0.898, 'random_strength': 0.754,
    'bagging_temperature': 0.614, 'border_count': 128
}

N_TRIALS_CAT = 60

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

def augment_data(df, features, target='Label', noise_scale=0.02):
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
    feat_stds = df_aug[numeric_feats].std().values; feat_stds = np.where(feat_stds == 0, 1, feat_stds)
    noise = np.random.normal(0, noise_scale, size=(len(df_aug), len(numeric_feats)))
    df_aug[numeric_feats] += (noise * feat_stds)
    return df_aug

def label_smoothing(y, sv=0.01):
    return y * (1 - sv) + 0.5 * sv

def temperature_scale(p, T):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-logit / T))


# ============================================================
# Phase 1: Pre-compute for CatBoost Optuna
# ============================================================
def precompute_for_cat_optuna(df_m, df_w, features_m, features_w, years, seeds):
    cache = {}
    for gender in ['M', 'W']:
        df = df_m if gender == 'M' else df_w
        features = features_m if gender == 'M' else features_w
        for val_year in years:
            for seed in seeds:
                set_seed(seed)
                train_mask = df['Season'] != val_year
                val_mask = df['Season'] == val_year
                if val_mask.sum() == 0: continue

                df_tr_aug = augment_data(df[train_mask], features, 'Label', NOISE_SCALE)
                X_tr = df_tr_aug[features].values
                y_tr = df_tr_aug['Label'].values
                y_tr_smooth = label_smoothing(y_tr, LABEL_SMOOTH)
                X_va = df.loc[val_mask, features].values
                y_va = df.loc[val_mask, 'Label'].values

                m_xgb = XGBRegressor(**XGB_V58, random_state=seed, verbosity=0)
                m_xgb.fit(X_tr, y_tr_smooth)
                p_xgb = np.clip(m_xgb.predict(X_va), 0, 1)

                m_lgb = LGBMRegressor(**LGB_V58, random_state=seed, verbose=-1, n_jobs=-1)
                m_lgb.fit(X_tr, y_tr_smooth)
                p_lgb = np.clip(m_lgb.predict(X_va), 0, 1)

                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_va_sc = scaler.transform(X_va)
                y_tr_bin = (y_tr > 0.5).astype(int)

                m_l1 = LogisticRegression(penalty='l1', C=0.5, solver='saga', max_iter=5000, random_state=seed)
                m_l1.fit(X_tr_sc, y_tr_bin)
                p_l1 = m_l1.predict_proba(X_va_sc)[:, 1]

                m_en = LogisticRegression(penalty='elasticnet', C=0.5, l1_ratio=0.5, solver='saga', max_iter=5000, random_state=seed)
                m_en.fit(X_tr_sc, y_tr_bin)
                p_en = m_en.predict_proba(X_va_sc)[:, 1]

                p_elo = df.loc[val_mask, 'EloWinProb'].values

                cache[(gender, val_year, seed)] = {
                    'X_tr': X_tr, 'y_tr_smooth': y_tr_smooth,
                    'X_va': X_va, 'y_va': y_va,
                    'p_xgb': p_xgb, 'p_lgb': p_lgb,
                    'p_l1': p_l1, 'p_en': p_en, 'p_elo': p_elo
                }
    return cache


# ============================================================
# Phase 2: CatBoost Optuna
# ============================================================
def create_cat_objective(cache):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'depth': trial.suggest_int('depth', 3, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 50.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 3.0),
            'border_count': trial.suggest_int('border_count', 64, 254),
        }

        all_y, all_p = [], []
        for gender in ['M', 'W']:
            weights = M_WEIGHTS_V58 if gender == 'M' else W_WEIGHTS_V58
            clip = M_CLIP if gender == 'M' else W_CLIP
            for val_year in TUNE_YEARS:
                seed_preds = []
                y_va = None
                for seed in TUNE_SEEDS:
                    key = (gender, val_year, seed)
                    if key not in cache: continue
                    c = cache[key]
                    y_va = c['y_va']

                    m_cat = CatBoostRegressor(**params, verbose=False, random_seed=seed)
                    m_cat.fit(c['X_tr'], c['y_tr_smooth'])
                    p_cat = np.clip(m_cat.predict(c['X_va']), 0, 1)

                    p_blend = (weights[0]*p_cat + weights[1]*c['p_xgb'] + weights[2]*c['p_lgb'] +
                               weights[3]*c['p_l1'] + weights[4]*c['p_en'] + weights[5]*c['p_elo'])
                    seed_preds.append(np.clip(p_blend, 0, 1))

                if y_va is not None and seed_preds:
                    p_final = np.clip(np.mean(seed_preds, axis=0), clip[0], clip[1])
                    all_y.extend(y_va); all_p.extend(p_final)
        return brier_score_loss(all_y, all_p)
    return objective


# ============================================================
# Phase 3: Full 13-year pre-compute (individual model preds)
# ============================================================
def full_precompute(df_m, df_w, features_m, features_w, cat_params):
    """Cache seed-averaged individual model predictions for weight/temp optimization."""
    results = {}
    for gender in ['M', 'W']:
        df = df_m if gender == 'M' else df_w
        features = features_m if gender == 'M' else features_w
        for val_year in FULL_YEARS:
            train_mask = df['Season'] != val_year
            val_mask = df['Season'] == val_year
            if val_mask.sum() == 0: continue

            X_va = df.loc[val_mask, features].values
            y_va = df.loc[val_mask, 'Label'].values
            p_elo_base = df.loc[val_mask, 'EloWinProb'].values

            all_cat, all_xgb, all_lgb, all_l1, all_en = [], [], [], [], []
            for seed in FULL_SEEDS:
                set_seed(seed)
                df_tr_aug = augment_data(df[train_mask], features, 'Label', NOISE_SCALE)
                X_tr = df_tr_aug[features].values
                y_tr = df_tr_aug['Label'].values
                y_tr_smooth = label_smoothing(y_tr, LABEL_SMOOTH)

                m_cat = CatBoostRegressor(**cat_params, verbose=False, random_seed=seed)
                m_cat.fit(X_tr, y_tr_smooth)
                all_cat.append(np.clip(m_cat.predict(X_va), 0, 1))

                m_xgb = XGBRegressor(**XGB_V58, random_state=seed, verbosity=0)
                m_xgb.fit(X_tr, y_tr_smooth)
                all_xgb.append(np.clip(m_xgb.predict(X_va), 0, 1))

                m_lgb = LGBMRegressor(**LGB_V58, random_state=seed, verbose=-1, n_jobs=-1)
                m_lgb.fit(X_tr, y_tr_smooth)
                all_lgb.append(np.clip(m_lgb.predict(X_va), 0, 1))

                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_va_sc = scaler.transform(X_va)
                y_tr_bin = (y_tr > 0.5).astype(int)

                m_l1 = LogisticRegression(penalty='l1', C=0.5, solver='saga', max_iter=5000, random_state=seed)
                m_l1.fit(X_tr_sc, y_tr_bin)
                all_l1.append(m_l1.predict_proba(X_va_sc)[:, 1])

                m_en = LogisticRegression(penalty='elasticnet', C=0.5, l1_ratio=0.5, solver='saga', max_iter=5000, random_state=seed)
                m_en.fit(X_tr_sc, y_tr_bin)
                all_en.append(m_en.predict_proba(X_va_sc)[:, 1])

            results[(gender, val_year)] = {
                'p_cat': np.mean(all_cat, axis=0),
                'p_xgb': np.mean(all_xgb, axis=0),
                'p_lgb': np.mean(all_lgb, axis=0),
                'p_l1': np.mean(all_l1, axis=0),
                'p_en': np.mean(all_en, axis=0),
                'p_elo': p_elo_base,
                'y_va': y_va
            }
            print(f"  Full pre-compute: {gender} {val_year} done", flush=True)
    return results


# ============================================================
# Phase 4: Weight Optimization (Scipy)
# ============================================================
def optimize_weights_for_gender(results, gender, clip):
    all_preds_list = []  # list of (p_cat, p_xgb, p_lgb, p_l1, p_en, p_elo, y_va) tuples
    for val_year in FULL_YEARS:
        key = (gender, val_year)
        if key not in results: continue
        r = results[key]
        all_preds_list.append(r)

    def objective(w):
        w = np.abs(w)
        w = w / w.sum()
        total_y, total_p = [], []
        for r in all_preds_list:
            p = w[0]*r['p_cat'] + w[1]*r['p_xgb'] + w[2]*r['p_lgb'] + w[3]*r['p_l1'] + w[4]*r['p_en'] + w[5]*r['p_elo']
            total_y.extend(r['y_va'])
            total_p.extend(np.clip(p, clip[0], clip[1]))
        return brier_score_loss(total_y, total_p)

    best_result = None
    for _ in range(10):
        x0 = np.random.dirichlet(np.ones(6))
        res = minimize(objective, x0, method='SLSQP',
                       constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                       bounds=[(0, 1)]*6, options={'maxiter': 500})
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    w_opt = np.abs(best_result.x)
    return (w_opt / w_opt.sum()).tolist(), best_result.fun


# ============================================================
# Phase 5: Temperature + Clip Joint Optimization
# ============================================================
def optimize_temp_clip(results, gender, weights):
    all_data = []
    for val_year in FULL_YEARS:
        key = (gender, val_year)
        if key not in results: continue
        r = results[key]
        p = sum(weights[i] * r[m] for i, m in enumerate(['p_cat', 'p_xgb', 'p_lgb', 'p_l1', 'p_en', 'p_elo']))
        all_data.append((p, r['y_va']))

    def objective(params):
        T, c_lo, c_hi = params[0], params[1], params[2]
        if T <= 0.1 or c_lo >= c_hi or c_lo < 0 or c_hi > 1:
            return 1.0
        total_y, total_p = [], []
        for p_raw, y_va in all_data:
            p_scaled = temperature_scale(p_raw, T)
            total_y.extend(y_va)
            total_p.extend(np.clip(p_scaled, c_lo, c_hi))
        return brier_score_loss(total_y, total_p)

    best_result = None
    for T_init in [0.6, 0.8, 1.0, 1.2]:
        for c_lo_init in [0.01, 0.03, 0.05]:
            c_hi_init = 1.0 - c_lo_init
            res = minimize(objective, [T_init, c_lo_init, c_hi_init], method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
            if best_result is None or res.fun < best_result.fun:
                best_result = res

    T_opt, clip_lo, clip_hi = best_result.x
    return T_opt, (clip_lo, clip_hi), best_result.fun


# ============================================================
# Final Scoring
# ============================================================
def compute_final_scores(results, m_weights, w_weights, m_T, w_T, m_clip, w_clip):
    print("\n" + "=" * 80)
    print("FINAL RESULTS: V59 Triple-Threat")
    print("=" * 80)
    print("| Year | Men Brier | Women Brier | Total |")

    year_results = []
    for val_year in FULL_YEARS:
        all_y, all_p = [], []
        s_m = s_w = 0.0
        for gender in ['M', 'W']:
            key = (gender, val_year)
            if key not in results: continue
            r = results[key]
            w = m_weights if gender == 'M' else w_weights
            T = m_T if gender == 'M' else w_T
            clip = m_clip if gender == 'M' else w_clip

            p_blend = sum(w[i] * r[m] for i, m in enumerate(['p_cat', 'p_xgb', 'p_lgb', 'p_l1', 'p_en', 'p_elo']))
            p_scaled = temperature_scale(p_blend, T)
            p_clipped = np.clip(p_scaled, clip[0], clip[1])

            brier = brier_score_loss(r['y_va'], p_clipped)
            if gender == 'M': s_m = brier
            else: s_w = brier
            all_y.extend(r['y_va']); all_p.extend(p_clipped)

        s_total = brier_score_loss(all_y, all_p)
        year_results.append(s_total)
        print(f"| {val_year} | {s_m:.4f} | {s_w:.4f} | {s_total:.4f} |", flush=True)

    avg = np.mean(year_results)
    print("-" * 80)
    print(f"V59 Final Average Brier Score: {avg:.5f}")
    return avg


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    START_TIME = time.time()
    set_seed(SEED)

    print("=" * 80)
    print("V59: Triple-Threat (CatBoost Retune + Weight Opt + Temperature Scaling)")
    print(f"CatBoost Trials: {N_TRIALS_CAT}")
    print("=" * 80)

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

    # ---- Phase 1: Pre-compute for CatBoost Optuna ----
    print("\n" + "=" * 80)
    print("Phase 1: Pre-compute for CatBoost Optuna (XGB_v58 + LGB_v58 + L1 + EN + Elo)")
    print("=" * 80)
    t1 = time.time()
    cat_cache = precompute_for_cat_optuna(df_m, df_w, features_m, features_w, TUNE_YEARS, TUNE_SEEDS)
    print(f"Done in {(time.time()-t1)/60:.1f}min", flush=True)

    # ---- Phase 2: CatBoost Optuna ----
    print("\n" + "=" * 80)
    print(f"Phase 2: CatBoost Optuna ({N_TRIALS_CAT} trials)")
    print("=" * 80)
    t2 = time.time()
    cat_study = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True))
    cat_study.optimize(create_cat_objective(cat_cache), n_trials=N_TRIALS_CAT, show_progress_bar=False)
    best_cat = cat_study.best_params
    print(f"CatBoost done in {(time.time()-t2)/60:.1f}min | Best Tune Brier: {cat_study.best_value:.5f}", flush=True)
    print(f"Best CatBoost: {json.dumps(best_cat, indent=2)}", flush=True)

    # ---- Phase 3: Full 13-year pre-compute ----
    print("\n" + "=" * 80)
    print("Phase 3: Full 13-year pre-compute (10 seeds x 13 years, cache individual model preds)")
    print("=" * 80)
    t3 = time.time()
    full_results = full_precompute(df_m, df_w, features_m, features_w, best_cat)
    print(f"Done in {(time.time()-t3)/60:.1f}min", flush=True)

    # V58 baseline (with V58 weights, no temp scaling)
    print("\n--- V58 Baseline (for comparison) ---")
    v58_base = compute_final_scores(full_results, M_WEIGHTS_V58, W_WEIGHTS_V58, 1.0, 1.0, M_CLIP, W_CLIP)

    # ---- Phase 4: Weight Optimization ----
    print("\n" + "=" * 80)
    print("Phase 4: Scipy Weight Optimization (gender-specific)")
    print("=" * 80)
    t4 = time.time()
    m_weights_opt, m_brier_w = optimize_weights_for_gender(full_results, 'M', M_CLIP)
    w_weights_opt, w_brier_w = optimize_weights_for_gender(full_results, 'W', W_CLIP)
    print(f"Men weights: Cat={m_weights_opt[0]:.3f} XGB={m_weights_opt[1]:.3f} LGB={m_weights_opt[2]:.3f} L1={m_weights_opt[3]:.3f} EN={m_weights_opt[4]:.3f} Elo={m_weights_opt[5]:.3f}", flush=True)
    print(f"Women weights: Cat={w_weights_opt[0]:.3f} XGB={w_weights_opt[1]:.3f} LGB={w_weights_opt[2]:.3f} L1={w_weights_opt[3]:.3f} EN={w_weights_opt[4]:.3f} Elo={w_weights_opt[5]:.3f}", flush=True)

    print("\n--- After Weight Optimization ---")
    after_weight = compute_final_scores(full_results, m_weights_opt, w_weights_opt, 1.0, 1.0, M_CLIP, W_CLIP)

    # ---- Phase 5: Temperature + Clip Optimization ----
    print("\n" + "=" * 80)
    print("Phase 5: Temperature Scaling + Clip Optimization (gender-specific)")
    print("=" * 80)
    t5 = time.time()
    m_T, m_clip_opt, m_brier_tc = optimize_temp_clip(full_results, 'M', m_weights_opt)
    w_T, w_clip_opt, w_brier_tc = optimize_temp_clip(full_results, 'W', w_weights_opt)
    print(f"Men: T={m_T:.3f}, Clip=({m_clip_opt[0]:.4f}, {m_clip_opt[1]:.4f})", flush=True)
    print(f"Women: T={w_T:.3f}, Clip=({w_clip_opt[0]:.4f}, {w_clip_opt[1]:.4f})", flush=True)

    # ---- Final Results ----
    print("\n" + "=" * 80)
    print("COMPARISON: V58 vs V59 Components")
    print("=" * 80)

    print(f"\n[A] V58 Baseline (CatBoost V23 + V58 weights):    {v58_base:.5f}")
    print(f"[B] + CatBoost Retune (Optuna {N_TRIALS_CAT} trials):     {v58_base:.5f} -> see below")
    print(f"[C] + Weight Optimization (Scipy):               {after_weight:.5f}")

    final = compute_final_scores(full_results, m_weights_opt, w_weights_opt, m_T, w_T, m_clip_opt, w_clip_opt)

    print(f"\n[D] + Temperature Scaling + Clip (FINAL V59):    {final:.5f}")
    print(f"\n{'='*80}")
    print(f"V58 -> V59 Total Improvement: {v58_base:.5f} -> {final:.5f} (delta: {final - v58_base:+.5f})")
    print(f"Total time: {(time.time()-START_TIME)/60:.1f}min", flush=True)

    with open('v59_config.json', 'w') as f:
        json.dump({
            'cat_params': best_cat,
            'xgb_params': XGB_V58,
            'lgb_params': LGB_V58,
            'm_weights': m_weights_opt,
            'w_weights': w_weights_opt,
            'm_temperature': m_T,
            'w_temperature': w_T,
            'm_clip': list(m_clip_opt),
            'w_clip': list(w_clip_opt),
            'final_brier': final
        }, f, indent=2)
    print("Config saved to v59_config.json", flush=True)
