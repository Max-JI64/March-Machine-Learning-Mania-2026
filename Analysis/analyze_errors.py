import pandas as pd

df = pd.read_csv('worst_predictions_v29.csv')

print("=== 분석: 오답 상위 100건 데이터의 평균 특성 ===")
print(f"가장 심하게 틀린 100경기 중...")
print(f"- 남성 대회(M) 비중: {len(df[df['League']=='M'])}건")
print(f"- 여성 대회(W) 비중: {len(df[df['League']=='W'])}건")

# 정배당팀(ELO가 훨씬 높은 팀)이 예상과 달리 패배한 "업셋" 경기 비율
# Label: 1이면 T1 승, 0이면 T2 승
# Predicted Prob이 아주 높은데 0이거나, 아주 낮은데 1인 경우
print("\n=== 오답 유형 분류 ===")
upsets = df[((df['Predicted_Prob'] > 0.8) & (df['Label'] == 0)) | ((df['Predicted_Prob'] < 0.2) & (df['Label'] == 1))]
print(f"극단적 업셋으로 인한 오답 (예측확률 80% 이상인데 패배 / 20% 이하인데 승리): {len(upsets)}건")

print("\n=== 오답이 많이 발생한 연도 ===")
print(df['Season'].value_counts().head(5))

print("\n=== 업셋 당한 강팀들의 특성 (T1/T2 불문, 졌지만 예상 확률が高았던 팀들) ===")
# 예상 확률과 실제 결과의 차이가 큰 데이터 탐색
# 주로 T1_Elo_Diff나 관련 지표의 평균 확인
print(f"평균 시드 격차 (SeedNum_Diff 절대값): {df['SeedNum_Diff'].abs().mean():.2f}")
print(f"평균 Elo 격차 (Elo_Diff 절대값): {df['Elo_Diff'].abs().mean():.2f}")
print(f"평균 3점슛 시도 비율 격차 (3PAr_Diff 절대값): {df['3PAr_Diff'].abs().mean():.4f}")
print(f"평균 페이스(Possessions) 격차 절대값: {df['Possessions_Diff'].abs().mean():.2f}")
print(f"평균 리바운드 확률(OR%) 격차 절대값: {df['OR%_Diff'].abs().mean():.4f}")

# 왜 이런 이변이 발생했는지에 대해 상관관계가 높은 변수를 찾아보자.
corr = df[['Brier_Loss', 'SeedNum_Diff', 'Elo_Diff', 'Possessions_Diff', '3PAr_Diff', 'OR%_Diff', 'Last30_WinRate_Diff', 'Margin_Std_Diff', 'TOV%_Diff', 'First_Round_Upset_Score_Diff']].abs().corr()
print("\n=== Brier_Loss 크기(얼마나 처참히 비껴갔는가)와의 상관관계 (절대값 비교) ===")
print(corr['Brier_Loss'].sort_values(ascending=False).head(6))
