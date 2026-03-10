import os
import glob
import time
import pandas as pd
import numpy as np
import random
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ==============================================================================
# 기본 설정
# ==============================================================================
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(SEED)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
START_TIME = time.time()
N_JOBS = min(os.cpu_count(), 64)

# ==============================================================================
# 데이터 로드 및 전처리 유틸리티
# ==============================================================================
def load_v11_features(gender='M', data_dir='../Data/preprocessed'):
    # V11에서 검증된 핵심 피처 리스트 로드
    with open('selected_features_v11.json', 'r') as f:
        selected_features = json.load(f)
    
    base_file = os.path.join(data_dir, f'base_matchup_features_{gender}.csv')
    df_main = pd.read_csv(base_file)
    
    # 모멘텀 피처 병합
    mom_file = os.path.join(data_dir, f'momentum_form_features_{gender}.csv')
    if os.path.exists(mom_file):
        mom_df = pd.read_csv(mom_file)
        for side in ['T1', 'T2']:
            df_main = pd.merge(df_main, mom_df, left_on=['Season', side], right_on=['Season', 'TeamID'], how='left').drop('TeamID', axis=1)
            df_main.rename(columns={c: f'{side}_{c}' for c in mom_df.columns if c not in ['Season', 'TeamID']}, inplace=True)
        for c in mom_df.columns:
            if c not in ['Season', 'TeamID']:
                df_main[f'{c}_Diff'] = df_main[f'T1_{c}'] - df_main[f'T2_{c}']
                
    # 필요한 Diff 컬럼 및 기초 컬럼만 유지
    keep_cols = ['Season', 'T1', 'T2', 'Label'] + selected_features
    # 존재하지 않는 컬럼 제외
    keep_cols = [c for c in keep_cols if c in df_main.columns]
    
    return df_main[keep_cols]

def augment_data(df, features, target='Label', noise_scale=0.02):
    df_swap = df.copy()
    diff_cols = [c for c in df.columns if c.endswith('_Diff') and c in features]
    for dc in diff_cols: df_swap[dc] = -df_swap[dc]
    df_swap[target] = 1 - df_swap[target]
    df_augmented = pd.concat([df, df_swap], ignore_index=True)
    return df_augmented

# ==============================================================================
# 신경망 모델 정의 (V9 기반)
# ==============================================================================
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def train_nn(X_train, y_train, X_val, input_dim):
    X_tr_t = torch.FloatTensor(X_train.values).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_train.values).reshape(-1, 1).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val.values).to(DEVICE)
    
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = SimpleNN(input_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    model.train()
    for _ in range(30): # Epochs
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy().flatten()
    return preds

# ==============================================================================
# 메인 실행 루프
# ==============================================================================
print("🚀 V12: Meta-Stacking Ensemble (CatBoost + XGBoost + NN)", flush=True)

df_m = load_v11_features('M')
df_w = load_v11_features('W')
common_cols = list(set(df_m.columns) & set(df_w.columns))
df_train = pd.concat([df_m[common_cols], df_w[common_cols]], ignore_index=True).fillna(0)

with open('selected_features_v11.json', 'r') as f:
    selected_features = json.load(f)
selected_features = [f for f in selected_features if f in df_train.columns]

full_val_seasons = sorted([int(s) for s in df_train['Season'].unique() if s >= 2013 and s != 2020])

oof_cat, oof_xgb, oof_nn, oof_targets = [], [], [], []
test_preds_list = [] # 실제 제출용이 아닌 검증용

print("\n🚀 베이스 모델 OOF 예측 수집 중...", flush=True)
print("| Year | CatBoost | XGBoost  | NeuralNet |", flush=True)

for val_year in full_val_seasons:
    train_mask = (df_train['Season'] < val_year)
    val_mask = (df_train['Season'] == val_year)
    
    X_tr = df_train.loc[train_mask, selected_features]
    y_tr = df_train.loc[train_mask, 'Label']
    X_val = df_train.loc[val_mask, selected_features]
    y_val = df_train.loc[val_mask, 'Label']
    
    # 1. CatBoost
    cat = CatBoostRegressor(iterations=1000, learning_rate=0.02, depth=6, verbose=False, thread_count=N_JOBS, random_seed=SEED)
    cat.fit(X_tr, y_tr)
    p_cat = np.clip(cat.predict(X_val), 0, 1)
    
    # 2. XGBoost
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, n_jobs=N_JOBS, random_state=SEED)
    xgb.fit(X_tr, y_tr)
    p_xgb = np.clip(xgb.predict(X_val), 0, 1)
    
    # 3. Neural Net
    p_nn = train_nn(X_tr, y_tr, X_val, len(selected_features))
    
    oof_cat.extend(p_cat)
    oof_xgb.extend(p_xgb)
    oof_nn.extend(p_nn)
    oof_targets.extend(y_val.values)
    
    print(f"| {val_year} | {brier_score_loss(y_val, p_cat):.4f}   | {brier_score_loss(y_val, p_xgb):.4f}   | {brier_score_loss(y_val, p_nn):.4f}    |", flush=True)

# ==============================================================================
# 메타 모델 학습 (Stacking)
# ==============================================================================
oof_meta_X = pd.DataFrame({
    'Cat': oof_cat,
    'XGB': oof_xgb,
    'NN': oof_nn
})

print("\n🚀 메타 모델(Logistic Regression) 학습 및 최종 평가...", flush=True)
meta_model = LogisticRegression(solver='lbfgs')
meta_model.fit(oof_meta_X, oof_targets)

final_preds = meta_model.predict_proba(oof_meta_X)[:, 1]
final_score = brier_score_loss(oof_targets, final_preds)

print("-" * 60, flush=True)
print(f"🎯 최종 V12 Stacking Brier Score: {final_score:.4f}", flush=True)
print(f"📊 메타 모델 가중치(Coefficients):")
for col, coef in zip(oof_meta_X.columns, meta_model.coef_[0]):
    print(f"   - {col}: {coef:.4f}")
print(f"⏱️ 총 소요 시간: {(time.time()-START_TIME)/60:.1f}분", flush=True)
print("-" * 60, flush=True)
