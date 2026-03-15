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

warnings.filterwarnings('ignore')

SEED = 42
DATA_DIR = '../Data/provided'
PREP_DIR = '../Data/preprocessed'
NOISE_SCALE = 0.02
SEEDS = [42, 123, 456, 789, 2024]
FULL_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

with open('v59_config.json') as f:
    v59 = json.load(f)
CAT_PARAMS = v59['cat_params']
XGB_PARAMS = v59['xgb_params']
LGB_PARAMS = v59['lgb_params']
LGB_PARAMS['objective'] = 'regression'

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

def train_and_eval_season_components(df_train, val_year, features):
    train_mask = df_train['Season'] != val_year
    val_mask = df_train['Season'] == val_year
    
    X_va = df_train.loc[val_mask, features].values
    y_va_label = df_train.loc[val_mask, 'Label'].values

    all_cat, all_xgb, all_lgb, all_ridge = [], [], [], []
    
    for seed in SEEDS:
        set_seed(seed)
        df_tr_aug = augment_data_margin(df_train[train_mask], features, 'Margin', noise_scale=NOISE_SCALE)
        X_tr = df_tr_aug[features].values
        y_tr = df_tr_aug['Margin'].values 
        
        m_cat = CatBoostRegressor(**CAT_PARAMS, verbose=False, random_seed=seed)
        m_cat.fit(X_tr, y_tr)
        all_cat.append(m_cat.predict(X_va))

        m_xgb = XGBRegressor(**XGB_PARAMS, random_state=seed, verbosity=0)
        m_xgb.fit(X_tr, y_tr)
        all_xgb.append(m_xgb.predict(X_va))

        m_lgb = LGBMRegressor(**LGB_PARAMS, random_state=seed, verbose=-1, n_jobs=-1)
        m_lgb.fit(X_tr, y_tr)
        all_lgb.append(m_lgb.predict(X_va))

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        m_ridge = Ridge(alpha=10.0, random_state=seed)
        m_ridge.fit(X_tr_sc, y_tr)
        all_ridge.append(m_ridge.predict(X_va_sc))

    return {
        'cat': np.mean(all_cat, axis=0),
        'xgb': np.mean(all_xgb, axis=0),
        'lgb': np.mean(all_lgb, axis=0),
        'ridge': np.mean(all_ridge, axis=0),
        'label': y_va_label
    }

if __name__ == "__main__":
    START_TIME = time.time()
    set_seed(SEED)

    print("=" * 80)
    print("V61: Margin Regression Ensemble Weight Optimization")
    print(f"Seeds: {len(SEEDS)} | Models: Cat, XGB, LGB, Ridge")
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

    print("\n" + "=" * 80)
    print("Phase 1: Generating Unweighted Margin Predictions (13 years)")
    print("=" * 80)
    
    preds_m = {}
    preds_w = {}
    
    for val_year in FULL_YEARS:
        t_start = time.time()
        print(f"  [Year {val_year}] Processing...", end=' ', flush=True)
        
        preds_m[val_year] = train_and_eval_season_components(df_m, val_year, features_m)
        if val_year in df_w['Season'].values:
            preds_w[val_year] = train_and_eval_season_components(df_w, val_year, features_w)
            
        print(f"Done in {(time.time() - t_start):.1f}s", flush=True)

    print("\n" + "=" * 80)
    print("Phase 2: Optimizing Weights & StdDev simultaneously via Scipy")
    print("=" * 80)
    
    def optimize_ensemble(preds_dict):
        all_cat, all_xgb, all_lgb, all_ridge, all_labels = [], [], [], [], []
        for year, data in preds_dict.items():
            all_cat.extend(data['cat'])
            all_xgb.extend(data['xgb'])
            all_lgb.extend(data['lgb'])
            all_ridge.extend(data['ridge'])
            all_labels.extend(data['label'])
            
        all_cat = np.array(all_cat); all_xgb = np.array(all_xgb); all_lgb = np.array(all_lgb); all_ridge = np.array(all_ridge); all_labels = np.array(all_labels)
        
        def objective(params):
            w_cat, w_xgb, w_lgb, w_ridge, std_dev = params
            # Normalize weights to sum to 1
            w_sum = w_cat + w_xgb + w_lgb + w_ridge
            if w_sum <= 0 or std_dev <= 0: return 1.0
            
            w_cat /= w_sum; w_xgb /= w_sum; w_lgb /= w_sum; w_ridge /= w_sum
            
            p_margin = (w_cat*all_cat + w_xgb*all_xgb + w_lgb*all_lgb + w_ridge*all_ridge)
            probs = convert_margin_to_prob(p_margin, std_dev)
            probs = np.clip(probs, 0.001, 0.999) 
            return brier_score_loss(all_labels, probs)
            
        best_res = None
        for std_init in [9.0, 10.5, 12.0]:
            x0 = [0.25, 0.25, 0.25, 0.25, std_init]
            bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (5.0, 20.0)]
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            if best_res is None or res.fun < best_res.fun:
                best_res = res
                
        opt_w = best_res.x[:4]
        opt_w = opt_w / opt_w.sum()
        opt_std = best_res.x[4]
        return opt_w, opt_std, best_res.fun

    m_w, m_std, m_brier = optimize_ensemble(preds_m)
    w_w, w_std, w_brier = optimize_ensemble(preds_w)
    
    print(f"Men Optimal Weights: Cat={m_w[0]:.3f}, XGB={m_w[1]:.3f}, LGB={m_w[2]:.3f}, Ridge={m_w[3]:.3f}")
    print(f"Men Optimal StdDev: {m_std:.3f} | Best Training Brier: {m_brier:.5f}\n")
    
    print(f"Women Optimal Weights: Cat={w_w[0]:.3f}, XGB={w_w[1]:.3f}, LGB={w_w[2]:.3f}, Ridge={w_w[3]:.3f}")
    print(f"Women Optimal StdDev: {w_std:.3f} | Best Training Brier: {w_brier:.5f}\n")
    
    print("\n" + "=" * 80)
    print("Phase 3: Final Brier Scores by Year (V61 Weighted)")
    print("=" * 80)
    print("| Year | Men Brier | Women Brier | Total |")
    
    total_y, total_p = [], []
    year_results = []
    
    for val_year in FULL_YEARS:
        # Men
        d_m = preds_m[val_year]
        p_margin_m = m_w[0]*d_m['cat'] + m_w[1]*d_m['xgb'] + m_w[2]*d_m['lgb'] + m_w[3]*d_m['ridge']
        p_prob_m = convert_margin_to_prob(p_margin_m, m_std)
        s_m = brier_score_loss(d_m['label'], p_prob_m)
        
        # Women
        if val_year in preds_w:
            d_w = preds_w[val_year]
            p_margin_w = w_w[0]*d_w['cat'] + w_w[1]*d_w['xgb'] + w_w[2]*d_w['lgb'] + w_w[3]*d_w['ridge']
            p_prob_w = convert_margin_to_prob(p_margin_w, w_std)
            s_w = brier_score_loss(d_w['label'], p_prob_w)
            y_comb = np.concatenate([d_m['label'], d_w['label']])
            p_comb = np.concatenate([p_prob_m, p_prob_w])
        else:
            s_w = 0.0
            y_comb = d_m['label']
            p_comb = p_prob_m
            
        s_total = brier_score_loss(y_comb, p_comb)
        year_results.append(s_total)
        total_y.extend(y_comb); total_p.extend(p_comb)
        print(f"| {val_year} | {s_m:.4f} | {s_w:.4f} | {s_total:.4f} |", flush=True)

    avg = np.mean(year_results)
    print("-" * 80)
    print(f"V60 Default Average Brier: 0.15893")
    print(f"V61 Final Average Brier Score: {avg:.5f} (delta: {avg - 0.15893:+.5f})")
    
    elapsed = (time.time() - START_TIME) / 60
    print(f"\nTotal time: {elapsed:.1f}min", flush=True)

    # Save configs
    with open('v61_config.json', 'w') as f:
        json.dump({
            'm_weights': m_w.tolist(),
            'w_weights': w_w.tolist(),
            'm_std': float(m_std),
            'w_std': float(w_std),
            'final_brier': float(avg)
        }, f, indent=2)
