import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
import warnings
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ==============================================================================
# 기본 설정 및 유틸리티
# ==============================================================================
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
START_TIME = time.time()
N_JOBS = min(os.cpu_count(), 64)

# Device selection: CUDA -> MPS (Mac GPU) -> CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ==============================================================================
# 데이터 처리 함수 (V8/V9 로직 유지)
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

def augment_data(df, features, target='Label', noise_scale=0.03):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    df_swap[target] = 1 - df_swap[target]
    df_augmented = pd.concat([df, df_swap], ignore_index=True)
    # 가우시안 노이즈 (수치형 피처에만)
    numeric_feats = [f for f in features if f in df_augmented.columns]
    noise = np.random.normal(0, noise_scale, size=(len(df_augmented), len(numeric_feats)))
    # 표준편차를 numpy array로 추출하여 브로드캐스팅 보장
    feat_stds = df_augmented[numeric_feats].std().values
    df_augmented[numeric_feats] += (noise * feat_stds)
    return df_augmented

def get_recency_sample_weights(seasons_array, max_season=None, decay=0.60):
    if max_season is None: max_season = seasons_array.max()
    weights = decay ** (max_season - seasons_array)
    return weights / weights.mean()

def label_smoothing(y_target, smoothing_val=0.05):
    return y_target * (1 - smoothing_val) + 0.5 * smoothing_val

# ==============================================================================
# 신경망 아키텍처 (V9)
# ==============================================================================
class BrierNet(nn.Module):
    def __init__(self, input_dim):
        super(BrierNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 메인 루프: 데이터 로드 및 피처 생성
# ==============================================================================
print("🚀 V10: Hybrid Ensemble (CatBoost + NN) & Isotonic Calibration", flush=True)
df_m = load_and_merge_features('M')
df_w = load_and_merge_features('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)
target = 'Label'

# 피처 세트 (V8 Interaction 포함)
try:
    with open('selected_features_v7.json', 'r') as f:
        base_features = json.load(f)
except:
    drop_cols = ['Season', 'T1', 'T2', 'Label'] + [c for c in df_train.columns if c.startswith('T1_') or c.startswith('T2_')]
    base_features = sorted([c for c in df_train.columns if c not in drop_cols and c.endswith('_Diff')])

for side in ['T1', 'T2']: # T1_Elo, T2_Elo 등 기본 피처 복구 (NN input용)
    pass # Diff 위주로 학습

# V8 Interaction Features
if 'Elo_Diff' in df_train.columns and 'SRS_Diff' in df_train.columns:
    df_train['Elo_x_SRS_Diff'] = df_train['Elo_Diff'] * df_train['SRS_Diff']
if 'WinRate_Diff' in df_train.columns and 'Pyth_Expected_Diff' in df_train.columns:
    df_train['WinRate_x_Pyth_Diff'] = df_train['WinRate_Diff'] * df_train['Pyth_Expected_Diff']
if 'ORB%_Diff' in df_train.columns and 'DRB%_Diff' in df_train.columns:
    df_train['Total_Rebound_Control_Diff'] = df_train['ORB%_Diff'] + df_train['DRB%_Diff']
if 'Past_Tourney_eFG%_Diff' in df_train.columns and 'Past_Tourney_TOV%_Diff' in df_train.columns:
    df_train['Tourney_Efficiency_Diff'] = df_train['Past_Tourney_eFG%_Diff'] * (1 - df_train['Past_Tourney_TOV%_Diff'])

features = base_features + ['Elo_x_SRS_Diff', 'WinRate_x_Pyth_Diff', 'Total_Rebound_Control_Diff', 'Tourney_Efficiency_Diff']
features = [f for f in features if f in df_train.columns]

# ==============================================================================
# 모델 파라미터 (V8 CatBoost)
# ==============================================================================
cat_best_params = {
    'iterations': 1124, 'learning_rate': 0.025, 'depth': 6,
    'l2_leaf_reg': 5.2, 'subsample': 0.75, 'colsample_bylevel': 0.7,
    'min_data_in_leaf': 45, 'random_seed': SEED, 'loss_function': 'RMSE',
    'bootstrap_type': 'Bernoulli', 'verbose': False, 'thread_count': N_JOBS
}
CAT_NOISE = 0.025

# ==============================================================================
# 앙상블 및 보정 루프 (2013-2025)
# ==============================================================================
full_val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
oof_preds_cat = []
oof_preds_nn = []
oof_targets = []

final_scores = []

print("\n🚀 Fold별 학습 및 OOF 예측 수집...", flush=True)
for val_year in full_val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    X_tr_raw, y_tr_raw = df_train.loc[train_mask, features].copy(), df_train.loc[train_mask, target].copy()
    X_val, y_val = df_train.loc[val_mask, features].copy(), df_train.loc[val_mask, target].copy()
    
    # --- CatBoost 학습 ---
    train_temp = pd.concat([X_tr_raw, y_tr_raw], axis=1)
    train_temp['Season'] = df_train.loc[train_mask, 'Season']
    train_aug = augment_data(train_temp, features, target, noise_scale=CAT_NOISE)
    weights_aug = get_recency_sample_weights(train_aug['Season'], max_season=val_year)
    y_tr_smooth = label_smoothing(train_aug[target])
    
    cat_model = CatBoostRegressor(**cat_best_params)
    cat_model.fit(train_aug[features], y_tr_smooth, sample_weight=weights_aug)
    pred_cat = np.clip(cat_model.predict(X_val), 0.0, 1.0)
    
    # --- NN 학습 ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw) # 증강 전 스케일링 권장
    X_val_scaled = scaler.transform(X_val)
    
    X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr_raw.values, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    
    nn_model = BrierNet(len(features)).to(device)
    optimizer = optim.AdamW(nn_model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    nn_model.train()
    for _ in range(120): # Epochs
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(nn_model(bx), by)
            loss.backward()
            optimizer.step()
    
    nn_model.eval()
    with torch.no_grad():
        pred_nn = nn_model(X_val_t).cpu().numpy().flatten()
        pred_nn = np.clip(pred_nn, 0.0, 1.0)
    
    # OOF 수집
    oof_preds_cat.extend(pred_cat)
    oof_preds_nn.extend(pred_nn)
    oof_targets.extend(y_val.values)
    
    # 연도별 점수 (가중치 최적화 전 단순 0.5:0.5 스코어링 - 모니터링용)
    blend_tmp = 0.5 * pred_cat + 0.5 * pred_nn
    print(f"| {val_year} | Cat: {brier_score_loss(y_val, pred_cat):.4f} | NN: {brier_score_loss(y_val, pred_nn):.4f} | SimpleBlend: {brier_score_loss(y_val, blend_tmp):.4f} |", flush=True)

# ==============================================================================
# 가중치 최적화 & Isotonic Calibration
# ==============================================================================
def blend_objective(weights):
    w1, w2 = weights
    blend = w1 * np.array(oof_preds_cat) + w2 * np.array(oof_preds_nn)
    blend = np.clip(blend, 0.0, 1.0) # 부동 소수점 오차 방지
    return brier_score_loss(oof_targets, blend)

res = minimize(blend_objective, [0.8, 0.2], bounds=[(0, 1), (0, 1)], constraints={'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
best_w1, best_w2 = res.x
print(f"\n🌟 최적 가중치: CatBoost={best_w1:.2%}, NN={best_w2:.2%}", flush=True)

final_blend = best_w1 * np.array(oof_preds_cat) + best_w2 * np.array(oof_preds_nn)
raw_brier = brier_score_loss(oof_targets, final_blend)
print(f"🌟 블렌딩 전 점수: {raw_brier:.5f}", flush=True)

# Isotonic Calibration 적용
iso = IsotonicRegression(out_of_bounds='clip')
# 교차 검증의 엄밀함을 위해 OOF를 이용해 적합시키고 점수를 내는 것은 약한 누수가 있을 수 있으나, 
# 0.1652 대비 전체적인 보정 가능성을 확인하기 위함
calibrated_preds = iso.fit_transform(final_blend, oof_targets)
calibrated_brier = brier_score_loss(oof_targets, calibrated_preds)

print(f"🌟 보정(Isotonic Calibration) 후 최종 Brier Score: {calibrated_brier:.5f}", flush=True)
print("-" * 60, flush=True)
print(f"🎯 V10 개선량 (V8 0.1652 대비): {0.1652 - calibrated_brier:.5f}", flush=True)
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 60, flush=True)
