import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 파트 I: Advanced Ratings 및 Feature Load (V8 로직 동일 유지)
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

# ==============================================================================
# 0. 증강 함수 및 기본 세팅
# ==============================================================================
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
START_TIME = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def augment_data(df, features, target='Label'):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    df_swap[target] = 1 - df_swap[target]
    df_augmented = pd.concat([df, df_swap], ignore_index=True)
    return df_augmented

# 데이터 로드
df_m = load_and_merge_features('M')
df_w = load_and_merge_features('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)
target = 'Label'

# ==============================================================================
# 2. V8: Feature Engineering 재적용
# ==============================================================================
try:
    with open('selected_features_v7.json', 'r') as f:
        base_features = json.load(f)
except FileNotFoundError:
    drop_cols = ['Season', 'T1', 'T2', 'Label', 'Is_Augmented'] + [c for c in df_train.columns if c.startswith('T1_') or c.startswith('T2_')]
    base_features = sorted([c for c in df_train.columns if c not in drop_cols and c.endswith('_Diff')])

new_features = []
if 'Elo_Diff' in df_train.columns and 'SRS_Diff' in df_train.columns:
    df_train['Elo_x_SRS_Diff'] = df_train['Elo_Diff'] * df_train['SRS_Diff']
    new_features.append('Elo_x_SRS_Diff')
if 'WinRate_Diff' in df_train.columns and 'Pyth_Expected_Diff' in df_train.columns:
    df_train['WinRate_x_Pyth_Diff'] = df_train['WinRate_Diff'] * df_train['Pyth_Expected_Diff']
    new_features.append('WinRate_x_Pyth_Diff')
if 'ORB%_Diff' in df_train.columns and 'DRB%_Diff' in df_train.columns:
    df_train['Total_Rebound_Control_Diff'] = df_train['ORB%_Diff'] + df_train['DRB%_Diff']
    new_features.append('Total_Rebound_Control_Diff')
if 'Past_Tourney_eFG%_Diff' in df_train.columns and 'Past_Tourney_TOV%_Diff' in df_train.columns:
    df_train['Tourney_Efficiency_Diff'] = df_train['Past_Tourney_eFG%_Diff'] * (1 - df_train['Past_Tourney_TOV%_Diff'])
    new_features.append('Tourney_Efficiency_Diff')

features = base_features + new_features

# ==============================================================================
# 3. PyTorch Neural Network 아키텍처 (V9)
# ==============================================================================
class BrierNet(nn.Module):
    def __init__(self, input_dim):
        super(BrierNet, self).__init__()
        # 극단적인 퍼포먼스를 위해 3-Layer Dense + Dropout 통과
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
            nn.Sigmoid() # 0 ~ 1 확률 출력
        )
    
    def forward(self, x):
        return self.net(x)

def smooth_labels(labels, smoothing=0.05):
    # 0 -> 0.05, 1 -> 0.95
    return labels * (1 - 2 * smoothing) + smoothing

EPOCHS = 150
BATCH_SIZE = 512
LR = 0.001

print(f"🚀 V9 극단적 신경망 아키텍처 학습 시작 | Device: {device}")
print("👉 Brier Score 0.1 이하 달성을 위한 강제 핏(Fit) 시도")

full_val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])
final_brier_per_year = []

for val_year in full_val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    X_tr_raw, y_tr_raw = df_train.loc[train_mask, features].copy(), df_train.loc[train_mask, target].copy()
    X_val, y_val = df_train.loc[val_mask, features].copy(), df_train.loc[val_mask, target].copy()
    
    # Data Augmentation (Swap)
    train_temp = pd.concat([X_tr_raw, y_tr_raw], axis=1)
    train_aug = augment_data(train_temp, features, target)
    X_tr = train_aug[features]
    y_tr = train_aug[target]
    
    # Scale Data (NN requires scaling)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    
    # Label Smoothing
    y_tr_smooth = smooth_labels(y_tr.values)
    
    # Convert to Tensors
    X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr_smooth, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(device)
    
    dataset = TensorDataset(X_tr_t, y_tr_t)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = BrierNet(input_dim=len(features)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = nn.MSELoss() # Brier Score와 동등한 MSELoss
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy().flatten()
        val_preds = np.clip(val_preds, 0.0, 1.0)
        score = brier_score_loss(y_val.values, val_preds)
        final_brier_per_year.append(score)
        print(f"| {val_year} | {score:.4f} |")

print("-" * 60)
print(f"🎯 최종 PyTorch V9(신경망) 평균 Brier Score: {np.mean(final_brier_per_year):.4f}")
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분")
print("-" * 60)
