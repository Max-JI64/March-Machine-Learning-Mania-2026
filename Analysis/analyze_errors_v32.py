import pandas as pd

df = pd.read_csv('worst_predictions_v32.csv')

print("=== 2차 분석(V32): 오답 상위 200건 데이터의 평균 특성 ===")
print(f"가장 심하게 틀린 200경기 중...")
print(f"- 남성 대회(M) 비중: {len(df[df['League']=='M'])}건")
print(f"- 여성 대회(W) 비중: {len(df[df['League']=='W'])}건")

print("\n=== 오답 유형 분류 ===")
upsets = df[((df['Predicted_Prob'] > 0.8) & (df['Label'] == 0)) | ((df['Predicted_Prob'] < 0.2) & (df['Label'] == 1))]
print(f"극단적 업셋으로 인한 오답 (예측확률 80% 이상인데 패배 / 20% 이하인데 승리): {len(upsets)}건")

# 이번에는 "기복(Margin_Std)"이나 "턴오버(TOV%)"가 아니라 아예 다른 지표들을 파고듭니다.
# 1. 모멘텀 (최근 14/30일 승률)
# 2. 휴식일 (마지막 경기 이후 쉰 날짜 차이)
# 3. 3점슛 의존도 (3PAr)
# 4. 공격 리바운드 (OR%)
# 5. 자유투 획득 비율 (FTr)

target_cols = [
    'Brier_Loss',
    'SeedNum_Diff',
    'Elo_Diff',
    'Last14_WinRate_Diff',
    'Last30_WinRate_Diff',
    'Days_Since_Last_Game_Diff',
    '3PAr_Diff',
    'OR%_Diff',
    'FTr_Diff',
    'TS%_Diff',
    'First_Round_Upset_Score_Diff'
]

# 존재하는 컬럼만 필터링
available_cols = [c for c in target_cols if c in df.columns]

if available_cols:
    corr = df[available_cols].abs().corr()
    print("\n=== 완전히 새로운(V32) 관점: Brier_Loss 크기와의 상관관계 (절대값) ===")
    print("기복이나 턴오버가 아닌 다른 이유(예: 모멘텀, 휴식일, 3점의존도 등)를 탐색합니다.")
    print(corr['Brier_Loss'].sort_values(ascending=False))
else:
    print("분석할 추가 피처가 부족합니다.")
    
print("\n=== 상위 50개 극단적 오답의 모멘텀/휴식일 평균 격차 ===")
top50 = df.head(50)
if 'Last14_WinRate_Diff' in top50.columns:
    print(f"평균 Last14_WinRate_Diff: {top50['Last14_WinRate_Diff'].mean():.4f}")
if 'Days_Since_Last_Game_Diff' in top50.columns:
    print(f"평균 Days_Since_Last_Game_Diff: {top50['Days_Since_Last_Game_Diff'].mean():.2f}")
if '3PAr_Diff' in top50.columns:
    print(f"평균 3PAr_Diff: {top50['3PAr_Diff'].mean():.4f}")
