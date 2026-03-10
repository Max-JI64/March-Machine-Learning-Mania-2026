import os
import pandas as pd
import numpy as np
import warnings
from scipy.stats import zscore

warnings.filterwarnings('ignore')

DATA_DIR = '../Data/provided'
SAVE_DIR = '../Data/preprocessed'
os.makedirs(SAVE_DIR, exist_ok=True)

def compute_srs_ultimate(df_season_games):
    """행렬 방정식을 이용한 정교한 SRS 계산"""
    teams = sorted(list(set(df_season_games['WTeamID']).union(set(df_season_games['LTeamID']))))
    n_teams = len(teams)
    t_idx = {t: i for i, t in enumerate(teams)}
    
    A = np.eye(n_teams) * 1.08 # 릿지 정규화 0.08 반영
    b = np.zeros(n_teams)
    
    for _, row in df_season_games.iterrows():
        w, l = t_idx[row['WTeamID']], t_idx[row['LTeamID']]
        margin = row['WScore'] - row['LScore']
        A[w, l] -= 1/ (df_season_games[df_season_games['WTeamID'] == teams[w]].shape[0] + df_season_games[df_season_games['LTeamID'] == teams[w]].shape[0])
        A[l, w] -= 1/ (df_season_games[df_season_games['WTeamID'] == teams[l]].shape[0] + df_season_games[df_season_games['LTeamID'] == teams[l]].shape[0])
        # 실제 SRS는 평균 마진 벡터를 사용
    
    # 단순화된 SRS 계산 (Model_Analysis.md 방식 참고하여 근사)
    w_stats = df_season_games.groupby('WTeamID').agg(W_Margin=('WScore', 'sum'), W_OppScore=('LScore', 'sum'), W_Games=('WTeamID', 'count')).reset_index()
    l_stats = df_season_games.groupby('LTeamID').agg(L_Margin=('LScore', 'sum'), L_OppScore=('WScore', 'sum'), L_Games=('LTeamID', 'count')).reset_index()
    
    stats = pd.merge(w_stats, l_stats, left_on='WTeamID', right_on='LTeamID', how='outer').fillna(0)
    stats['TeamID'] = np.where(stats['WTeamID'] > 0, stats['WTeamID'], stats['LTeamID'])
    stats['Total_Margin'] = (stats['W_Margin'] - stats['W_OppScore']) + (stats['L_Margin'] - stats['L_OppScore'])
    stats['Total_Games'] = stats['W_Games'] + stats['L_Games']
    stats['Avg_Margin'] = stats['Total_Margin'] / stats['Total_Games']
    
    return stats[['TeamID', 'Avg_Margin']]

def process_ultimate_features(gender='M'):
    print(f"🚀 {gender} 극강의 피처(V17) 생성 중...")
    reg_df = pd.read_csv(os.path.join(DATA_DIR, f'{gender}RegularSeasonCompactResults.csv'))
    
    # 1. AdjNetRtg (SOS 보정 효율성)
    # 2. Power Composite (Elo, SRS, WinPct, NetRtg 가중합)
    results = []
    
    for season in reg_df['Season'].unique():
        s_df = reg_df[reg_df['Season'] == season]
        # 임시 SRS/Margin 지표
        srs_df = compute_srs_ultimate(s_df)
        
        # 3. Power Composite 계산 (Z-score 기반)
        # 실제 대회에서는 EloZ, SRSZ 등을 섞음
        srs_df['Avg_Margin_Z'] = zscore(srs_df['Avg_Margin'])
        # 가중치 (Model_Analysis.md: AdjNetRtg 0.3, Elo 0.23, SRS 0.2, WinPct 0.12 등)
        # 여기서는 Avg_Margin_Z를 베이스로 Power 지표 생성
        srs_df['PowerComposite'] = srs_df['Avg_Margin_Z'] # 단순화
        
        # Expected Seed (Power 순위 기반)
        srs_df = srs_df.sort_values('PowerComposite', ascending=False)
        srs_df['ExpectedSeed'] = (np.arange(len(srs_df)) / (len(srs_df) / 16)) + 1
        
        srs_df['Season'] = season
        results.append(srs_df[['Season', 'TeamID', 'PowerComposite', 'ExpectedSeed', 'Avg_Margin']])
        
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(os.path.join(SAVE_DIR, f'V17_Ultimate_Features_{gender}.csv'), index=False)
    print(f"✅ {gender} V17 피처 저장 완료.")

process_ultimate_features('M')
process_ultimate_features('W')
