import os, glob, time, json, random, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
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
LABEL_SMOOTH = 0.01
NOISE_SCALE = 0.02

TUNE_SEEDS = [42, 123]
FULL_SEEDS = [42, 123, 456, 789, 2024, 7, 31, 99, 555, 1234]
TUNE_YEARS = [2023, 2024, 2025]
FULL_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

M_WEIGHTS = [0.16, 0.30, 0.14, 0.15, 0.15, 0.10]
W_WEIGHTS = [0.30, 0.14, 0.14, 0.15, 0.15, 0.12]
M_CLIP, W_CLIP = (0.05, 0.95), (0.01, 0.99)

CAT_PARAMS = {
    'iterations': 1037, 'learning_rate': 0.0292, 'depth': 6,
    'l2_leaf_reg': 10.09, 'subsample': 0.898, 'random_strength': 0.754,
    'bagging_temperature': 0.614, 'border_count': 128
}
XGB_DEFAULT = {'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 5, 'subsample': 0.8}
LGB_DEFAULT = {'n_estimators': 1000, 'learning_rate': 0.01, 'num_leaves': 31, 'subsample': 0.8}

N_TRIALS_XGB = 40
N_TRIALS_LGB = 40

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


# ============================================================
# Phase 1: Pre-compute fixed model predictions
# ============================================================
def precompute_cache(df_m, df_w, features_m, features_w, years, seeds):
    cache = {}
    total = len(years) * len(seeds) * 2
    done = 0
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

                m_cat = CatBoostRegressor(**CAT_PARAMS, verbose=False, random_seed=seed)
                m_cat.fit(X_tr, y_tr_smooth)
                p_cat = np.clip(m_cat.predict(X_va), 0, 1)

                m_xgb = XGBRegressor(**XGB_DEFAULT, random_state=seed, verbosity=0)
                m_xgb.fit(X_tr, y_tr_smooth)
                p_xgb_def = np.clip(m_xgb.predict(X_va), 0, 1)

                m_lgb = LGBMRegressor(**LGB_DEFAULT, random_state=seed, verbose=-1, n_jobs=-1)
                m_lgb.fit(X_tr, y_tr_smooth)
                p_lgb_def = np.clip(m_lgb.predict(X_va), 0, 1)

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
                    'p_cat': p_cat, 'p_xgb': p_xgb_def, 'p_lgb': p_lgb_def,
                    'p_l1': p_l1, 'p_en': p_en, 'p_elo': p_elo
                }
                done += 1
                print(f"  Pre-compute [{done}/{total}] {gender} {val_year} seed={seed}", flush=True)
    return cache


def compute_blend_brier(cache, years, seeds, replace_model=None, replace_params=None):
    """Compute blended Brier using cache. Optionally replace one model's predictions."""
    all_y, all_p = [], []
    for gender in ['M', 'W']:
        weights = M_WEIGHTS if gender == 'M' else W_WEIGHTS
        clip = M_CLIP if gender == 'M' else W_CLIP
        for val_year in years:
            seed_preds = []
            y_va = None
            for seed in seeds:
                key = (gender, val_year, seed)
                if key not in cache: continue
                c = cache[key]
                y_va = c['y_va']

                if replace_model == 'xgb' and replace_params is not None:
                    m = XGBRegressor(**replace_params, random_state=seed, verbosity=0)
                    m.fit(c['X_tr'], c['y_tr_smooth'])
                    p_xgb = np.clip(m.predict(c['X_va']), 0, 1)
                else:
                    p_xgb = c['p_xgb']

                if replace_model == 'lgb' and replace_params is not None:
                    m = LGBMRegressor(**replace_params, random_state=seed, verbose=-1, n_jobs=-1)
                    m.fit(c['X_tr'], c['y_tr_smooth'])
                    p_lgb = np.clip(m.predict(c['X_va']), 0, 1)
                else:
                    p_lgb = c['p_lgb']

                p_blend = (weights[0]*c['p_cat'] + weights[1]*p_xgb + weights[2]*p_lgb +
                           weights[3]*c['p_l1'] + weights[4]*c['p_en'] + weights[5]*c['p_elo'])
                seed_preds.append(np.clip(p_blend, 0, 1))

            if y_va is not None and len(seed_preds) > 0:
                p_final = np.clip(np.mean(seed_preds, axis=0), clip[0], clip[1])
                all_y.extend(y_va)
                all_p.extend(p_final)
    return brier_score_loss(all_y, all_p)


# ============================================================
# Phase 2 & 3: Optuna Objectives
# ============================================================
def create_xgb_objective(cache):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        }
        return compute_blend_brier(cache, TUNE_YEARS, TUNE_SEEDS, 'xgb', params)
    return objective

def create_lgb_objective(cache):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 10, 63),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'objective': trial.suggest_categorical('objective', ['regression', 'cross_entropy']),
        }
        return compute_blend_brier(cache, TUNE_YEARS, TUNE_SEEDS, 'lgb', params)
    return objective


def update_cache_model(cache, model_name, best_params):
    """Re-compute a model's predictions with optimal params and update cache."""
    for key in cache:
        gender, val_year, seed = key
        c = cache[key]
        if model_name == 'xgb':
            m = XGBRegressor(**best_params, random_state=seed, verbosity=0)
            m.fit(c['X_tr'], c['y_tr_smooth'])
            c['p_xgb'] = np.clip(m.predict(c['X_va']), 0, 1)
        elif model_name == 'lgb':
            m = LGBMRegressor(**best_params, random_state=seed, verbose=-1, n_jobs=-1)
            m.fit(c['X_tr'], c['y_tr_smooth'])
            c['p_lgb'] = np.clip(m.predict(c['X_va']), 0, 1)


# ============================================================
# Phase 4: Final Full Evaluation
# ============================================================
def final_evaluation(df_m, df_w, features_m, features_w, best_xgb, best_lgb):
    print("\n" + "=" * 80)
    print("Phase 4: Final Full Evaluation (10 seeds x 13 years)")
    print("=" * 80)
    print("| Year | Men Brier | Women Brier | Total |")

    results = []
    for val_year in FULL_YEARS:
        all_y, all_p = [], []
        s_m = s_w = 0.0

        for gender in ['M', 'W']:
            df = df_m if gender == 'M' else df_w
            features = features_m if gender == 'M' else features_w
            weights = M_WEIGHTS if gender == 'M' else W_WEIGHTS
            clip = M_CLIP if gender == 'M' else W_CLIP

            train_mask = df['Season'] != val_year
            val_mask = df['Season'] == val_year
            if val_mask.sum() == 0: continue

            X_va = df.loc[val_mask, features].values
            y_va = df.loc[val_mask, 'Label'].values

            seed_preds = []
            for seed in FULL_SEEDS:
                set_seed(seed)
                df_tr_aug = augment_data(df[train_mask], features, 'Label', NOISE_SCALE)
                X_tr = df_tr_aug[features].values
                y_tr = df_tr_aug['Label'].values
                y_tr_smooth = label_smoothing(y_tr, LABEL_SMOOTH)

                m_cat = CatBoostRegressor(**CAT_PARAMS, verbose=False, random_seed=seed)
                m_cat.fit(X_tr, y_tr_smooth)
                p_cat = np.clip(m_cat.predict(X_va), 0, 1)

                m_xgb = XGBRegressor(**best_xgb, random_state=seed, verbosity=0)
                m_xgb.fit(X_tr, y_tr_smooth)
                p_xgb = np.clip(m_xgb.predict(X_va), 0, 1)

                m_lgb = LGBMRegressor(**best_lgb, random_state=seed, verbose=-1, n_jobs=-1)
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

                p_blend = (weights[0]*p_cat + weights[1]*p_xgb + weights[2]*p_lgb +
                           weights[3]*p_l1 + weights[4]*p_en + weights[5]*p_elo)
                seed_preds.append(np.clip(p_blend, 0, 1))

            p_final = np.clip(np.mean(seed_preds, axis=0), clip[0], clip[1])
            brier = brier_score_loss(y_va, p_final)
            if gender == 'M': s_m = brier
            else: s_w = brier
            all_y.extend(y_va); all_p.extend(p_final)

        s_total = brier_score_loss(all_y, all_p)
        results.append(s_total)
        print(f"| {val_year} | {s_m:.4f} | {s_w:.4f} | {s_total:.4f} |", flush=True)

    avg = np.mean(results)
    print("-" * 80)
    print(f"V58 Final Average Brier Score: {avg:.5f}")
    return avg, results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    START_TIME = time.time()
    set_seed(SEED)

    print("=" * 80)
    print("V58: Optuna Re-tuning XGBoost & LightGBM (V55 Leakage-Free Pipeline)")
    print(f"XGB Trials: {N_TRIALS_XGB} | LGB Trials: {N_TRIALS_LGB}")
    print(f"Tune: {len(TUNE_SEEDS)} seeds x {len(TUNE_YEARS)} years | Final: {len(FULL_SEEDS)} seeds x {len(FULL_YEARS)} years")
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

    # Phase 1: Pre-compute
    print("\n" + "=" * 80)
    print("Phase 1: Pre-computing CatBoost/L1/EN/Elo + default XGB/LGB predictions")
    print("=" * 80)
    t1 = time.time()
    cache = precompute_cache(df_m, df_w, features_m, features_w, TUNE_YEARS, TUNE_SEEDS)
    print(f"Pre-compute done in {(time.time()-t1)/60:.1f}min ({len(cache)} combos)", flush=True)

    baseline = compute_blend_brier(cache, TUNE_YEARS, TUNE_SEEDS)
    print(f"Default V55 Tune Brier (2 seeds x 3 years): {baseline:.5f}", flush=True)

    # Phase 2: Optuna XGBoost
    print("\n" + "=" * 80)
    print(f"Phase 2: Optuna XGBoost ({N_TRIALS_XGB} trials)")
    print("=" * 80)
    t2 = time.time()
    xgb_study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    )
    xgb_study.optimize(create_xgb_objective(cache), n_trials=N_TRIALS_XGB, show_progress_bar=False)

    best_xgb = xgb_study.best_params
    print(f"XGB done in {(time.time()-t2)/60:.1f}min | Best Brier: {xgb_study.best_value:.5f}", flush=True)
    print(f"Best XGB: {json.dumps(best_xgb, indent=2)}", flush=True)

    # Update cache with optimal XGB before LGB tuning
    print("Updating cache with optimal XGB...", flush=True)
    update_cache_model(cache, 'xgb', best_xgb)

    post_xgb = compute_blend_brier(cache, TUNE_YEARS, TUNE_SEEDS)
    print(f"After XGB Opt Tune Brier: {post_xgb:.5f} (was {baseline:.5f})", flush=True)

    # Phase 3: Optuna LightGBM
    print("\n" + "=" * 80)
    print(f"Phase 3: Optuna LightGBM ({N_TRIALS_LGB} trials)")
    print("=" * 80)
    t3 = time.time()
    lgb_study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    )
    lgb_study.optimize(create_lgb_objective(cache), n_trials=N_TRIALS_LGB, show_progress_bar=False)

    best_lgb = lgb_study.best_params
    print(f"LGB done in {(time.time()-t3)/60:.1f}min | Best Brier: {lgb_study.best_value:.5f}", flush=True)
    print(f"Best LGB: {json.dumps(best_lgb, indent=2)}", flush=True)

    # Summary
    print("\n" + "=" * 80)
    print("Optuna Summary")
    print("=" * 80)
    print(f"Default V55 Tune Brier:     {baseline:.5f}")
    print(f"After XGB Optimization:     {xgb_study.best_value:.5f} (delta: {xgb_study.best_value - baseline:+.5f})")
    print(f"After XGB+LGB Optimization: {lgb_study.best_value:.5f} (delta: {lgb_study.best_value - baseline:+.5f})")

    # Phase 4: Final full evaluation
    avg, results = final_evaluation(df_m, df_w, features_m, features_w, best_xgb, best_lgb)

    elapsed = (time.time() - START_TIME) / 60
    print(f"\nTotal time: {elapsed:.1f}min", flush=True)

    with open('v58_best_params.json', 'w') as f:
        json.dump({'xgb': best_xgb, 'lgb': best_lgb, 'final_brier': avg}, f, indent=2)
    print("Params saved to v58_best_params.json", flush=True)
