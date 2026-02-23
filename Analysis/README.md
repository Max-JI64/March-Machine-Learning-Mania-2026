# March Machine Learning Mania 2026 - 분석 및 모델링 기본 골자

본 문서는 캐글(Kaggle) "March Machine Learning Mania 2026" 대회의 데이터를 분석하고 예측 모델을 구축하기 위한 핵심 전략과 파이프라인 구조를 정리한 것입니다.

## 1. 대회 핵심 목표 (Objective)
*   **과제:** 과거에 한 번도 맞붙은 적 없는 팀들을 포함하여, 토너먼트에 진출 가능한 모든 팀 간의 1:1 매치업에 대해 승리 확률 예측.
*   **평가 지표 (Metric):** **Brier Score** (이 맥락에서는 **MSE, 평균 제곱 오차**와 동일).
    *   예측 확률과 실제 결과(승리 1, 패배 0) 차이의 제곱 평균을 최소화하는 것이 핵심 목표입니다.
    *   극단적인 예측(예: 1.0 또는 0.0)이 틀렸을 때 받는 페널티에 대응하기 위해 도입된 평가지표입니다. 모델의 평가/검증 지표(Validation Metric)를 반드시 Brier Score(또는 MSE)로 설정하여 학습시켜야 합니다.

---

## 2. 정답 데이터 (Target Variable) 정의
대회측에서 제공하는 `SampleSubmission~.csv` 형식으로 정답을 만들 필요가 없습니다. 정답은 이미 과거의 경기 결과 파일들에 포함되어 있습니다.

*   **사용 데이터:** `MRegularSeasonCompactResults.csv`, `MNCAATourneyCompactResults.csv` 등
*   **Target 생성 방식:**
    1.  각 경기 데이터에는 승리팀(`WTeamID`)과 패배팀(`LTeamID`)이 기록되어 있습니다.
    2.  모델이 특정 위치에 편향되지 않도록 두 팀의 순서를 무작위로 섞어 `Team_A`, `Team_B` 형태로 재구성합니다.
    3.  `Team_A`가 승리한 경우 `Target = 1`, 패배한 경우 `Target = 0`으로 레이블링합니다.

---

## 3. 피처 엔지니어링 (Feature Engineering)
모델이 처음 보는(맞붙은 적 없는) 두 팀의 승패를 예측하려면, "팀의 고유 ID"가 아니라 **"팀의 전력 차원(스탯)"**을 학습시켜야 합니다.

*   **팀별 기본 능력치 요약:**
    *   정규 시즌 데이터를 통해 승률, 평균 득/실점, 리바운드, 어시스트, 시드 번호, 순위(Massey Ordinals) 등의 시즌 통계를 집계합니다.
*   **매치업 피처(차이값) 계산:**
    *   대규모 매치업을 수치화하기 위해 `Team_A` 통계와 `Team_B` 통계의 **차이(Difference)** 혹은 **비율(Ratio)**을 계산합니다.
    *   *예시:* `Diff_승률 = Team_A_승률 - Team_B_승률`, `Diff_시드 = Team_A_시드 - Team_B_시드`
    *   이러한 차이값 특성들이 모델에게 "어느 정도 전력 차이가 날 때 누가 이기는가"에 대한 일반적인 패턴을 학습하게 합니다.

---

## 4. 학습 전략 (Training Strategy)
*   **데이터 분할 (Data Splitting):**
    *   과거 토너먼트 데이터를 연도별로 분할하여 Train/Validation 셋을 구성합니다. (예: 2022년 이전 데이터로 훈련, 2023~2024년 데이터로 검증)
*   **모델링 (Modeling):**
    *   **트리 기반 모델 사용 시 (XGBoost, LightGBM 등):** 기본 분류 손실 함수(Log Loss 등)를 사용하더라도 검증 지표를 `MSE` 또는 `Brier Score`로 설정하여 Early Stopping을 적용합니다.
    *   **분포 및 확률 보정 (Calibration):** 트리기반 모델은 확률값을 넓게 퍼뜨리지 못할 수 있으므로, 최종 예측 후 Isotonic Regression이나 Platt Scaling 기법으로 예측 확률(Pred)을 튜닝하는 과정이 Brier 점수 향상에 유리할 수 있습니다.

---

## 5. 예측 및 제출 (Prediction & Submission)
2026년 대회가 실제로 개최되어 대진 예측을 할 때의 파이프라인입니다.

1.  2026년 정규 시즌 성적을 바탕으로 출전한 모든 팀의 스탯을 갱신 및 계산.
2.  `SampleSubmissionStage1.csv` 및 `SampleSubmissionStage2.csv`에 적힌 모든 가상 매치업(`ID: 2026_1101_1102`)을 가져옴.
3.  매치업된 두 팀의 스탯 차이 Feature를 생성.
4.  훈련된 모델에 통과시켜 각 매치업의 승리 예측 확률(Pred) 도출.
5.  형식에 맞춰 제출(Submission) 파일 생성.
