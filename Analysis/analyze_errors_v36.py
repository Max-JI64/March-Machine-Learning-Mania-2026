import pandas as pd
import numpy as np

df = pd.read_csv('worst_predictions_v35.csv')

print("=== 3차 분석(V36): V35 마스터 모델 오답 상위 200건 데이터의 평균 특성 ===")
print(f"가장 심하게 틀린 200경기 중...")
print(f"- 남성 대회(M) 비중: {len(df[df['League']=='M'])}건")
print(f"- 여성 대회(W) 비중: {len(df[df['League']=='W'])}건")

print("\n=== 오답 유형 분류 (극단적 업셋) ===")
upsets = df[((df['Predicted_Prob'] > 0.8) & (df['Label'] == 0)) | ((df['Predicted_Prob'] < 0.2) & (df['Label'] == 1))]
print(f"예측확률 80% 이상인데 패배 / 20% 이하인데 승리: {len(upsets)}건")

# 이번에는 "모멘텀(Last30)", "DNA(Upset_Score)"가 추가된 V35에서도 잡히지 않은 미지의 영역을 찾습니다.
# 주요 가설
# 1. 특정 플레이스타일 상성 (예: 3점을 많이 쏘는 팀 vs 3점 수비가 약한 팀)
# 2. 자유투 관련 요소 (접전 상황에서의 자유투 정확도)
# 3. 신장/리바운드 마진 (압도적인 높이 차이가 이변을 만듦)

target_cols = [
    'Brier_Loss',
    'SeedNum_Diff',
    'Elo_Diff',
    '3PAr_Diff',
    'OR%_Diff',
    'FTr_Diff',
    'TS%_Diff',
    'OffRtg_Diff',
    'DefRtg_Diff',
    'AstTO_Ratio_Diff',
    'Possessions_mean_Diff' # 페이스/템포의 차이 (빠른 팀 vs 느린 팀)
]

available_cols = [c for c in target_cols if c in df.columns]

if available_cols:
    corr = df[available_cols].abs().corr()
    print("\n=== Brier_Loss 크기와의 상관관계 (절대값) ===")
    print("기존의 기복, 턴오버, 모멘텀을 제외하고 어떤 플레이스타일 지표가 업셋을 유발했는지 봅니다.")
    print(corr['Brier_Loss'].sort_values(ascending=False))
else:
    print("분석할 추가 피처가 없습니다.")

print("\n=== 상위 50개 극단적 오답의 플레이스타일(Style) 평균 격차 ===")
top50 = df.head(50)
if '3PAr_Diff' in top50.columns:
    print(f"평균 3PAr_Diff (3점 의존도 차이): {top50['3PAr_Diff'].mean():.4f}")
if 'OR%_Diff' in top50.columns:
    print(f"평균 OR%_Diff (공격 리바운드 차이): {top50['OR%_Diff'].mean():.4f}")
if 'AstTO_Ratio_Diff' in top50.columns:
    print(f"평균 AstTO_Ratio_Diff (어시/턴오버 안정감 차이): {top50['AstTO_Ratio_Diff'].mean():.4f}")
if 'OffRtg_Diff' in top50.columns:
    print(f"평균 OffRtg_Diff (공격 효율성 차이): {top50['OffRtg_Diff'].mean():.4f}")
if 'DefRtg_Diff' in top50.columns:
    print(f"평균 DefRtg_Diff (수비 효율성 차이): {top50['DefRtg_Diff'].mean():.4f}")

# 상성 팩터 생성 (가상 테스트)
if 'OffRtg_Diff' in df.columns and 'DefRtg_Diff' in df.columns:
    # 공격효율이 너무 좋은데 수비효율이 엉망인 '유리대포' 팀 리스크
    df['Glass_Cannon_Risk'] = df['OffRtg_Diff'] * df['DefRtg_Diff']
    print(f"\n[실험] Glass Cannon Risk와 Brier Loss 상관관계: {df['Brier_Loss'].corr(df['Glass_Cannon_Risk'].abs()):.4f}")
