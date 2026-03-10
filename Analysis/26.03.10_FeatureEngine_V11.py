import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# ==============================================================================
# 기본 경로 설정
# ==============================================================================
DATA_DIR = '../Data/provided'
PREP_DIR = '../Data/preprocessed'

def create_rolling_features(gender='M'):
    print(f"🚀 {gender} 정규 시즌 롤링(Rolling) 피처 및 모멘텀 지표 생성 중...")
    
    # 1. 원본 데이터 로드
    reg_df = pd.read_csv(os.path.join(DATA_DIR, f'{gender}RegularSeasonCompactResults.csv'))
    
    # 2. 팀별/경기별 데이터로 재편성 (W/L 관점 -> Team 관점)
    # 승리팀 관점
    w_df = reg_df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LScore']].rename(
        columns={'WTeamID': 'TeamID', 'WScore': 'Score', 'LScore': 'OppScore'}
    )
    w_df['IsWin'] = 1
    
    # 패배팀 관점
    l_df = reg_df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WScore']].rename(
        columns={'LTeamID': 'TeamID', 'LScore': 'Score', 'WScore': 'OppScore'}
    )
    l_df['IsWin'] = 0
    
    # 통합 및 정렬
    team_games = pd.concat([w_df, l_df], ignore_index=True).sort_values(['Season', 'TeamID', 'DayNum'])
    team_games['Margin'] = team_games['Score'] - team_games['OppScore']
    
    momentum_records = []
    
    # 3. 롤링 지표 계산 (최근 14일, 30일)
    for (season, team_id), group in tqdm(team_games.groupby(['Season', 'TeamID'])):
        group = group.reset_index(drop=True)
        
        # 시즌 마감일 기준 (보통 DayNum 132)
        # 본 대회 규정상 토너먼트 직전 상태를 고정해야 하므로, 132일 기준의 최종 모멘텀을 추출
        # 하지만 학습 데이터(과거 토너먼트)를 위해 각 경기 시점의 모멘텀을 구하는 것이 아니라,
        # 각 시즌의 정규시즌 종료 시점 지표를 해당 시즌의 모든 토너먼트 매치업에 일괄 적용함.
        
        # 최근 14일 (DayNum 118 ~ 132)
        last14 = group[group['DayNum'] > 118]
        # 최근 30일 (DayNum 102 ~ 132)
        last30 = group[group['DayNum'] > 102]
        
        # 스트릭(Streak) 계산 (마지막 경기부터 역순으로)
        streak = 0
        results = group['IsWin'].values
        if len(results) > 0:
            last_res = results[-1]
            for r in reversed(results):
                if r == last_res:
                    streak += 1 if last_res == 1 else -1
                else:
                    break
        
        res = {
            'Season': season,
            'TeamID': team_id,
            'Last14_WinRate': last14['IsWin'].mean() if len(last14) > 0 else 0.5,
            'Last14_Margin': last14['Margin'].mean() if len(last14) > 0 else 0,
            'Last30_WinRate': last30['IsWin'].mean() if len(last30) > 0 else 0.5,
            'Last30_Margin': last30['Margin'].mean() if len(last30) > 0 else 0,
            'Active_Streak': streak
        }
        momentum_records.append(res)
    
    momentum_df = pd.DataFrame(momentum_records)
    output_file = os.path.join(PREP_DIR, f'momentum_form_features_{gender}.csv')
    momentum_df.to_csv(output_file, index=False)
    print(f"✅ {output_file} 저장 완료!")

if __name__ == "__main__":
    if not os.path.exists(PREP_DIR):
        os.makedirs(PREP_DIR)
    
    create_rolling_features('M')
    create_rolling_features('W')
    print("\n🚀 V11 피처 엔지니어링 1단계(모멘텀) 완료!")
