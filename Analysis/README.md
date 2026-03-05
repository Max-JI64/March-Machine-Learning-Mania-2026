# March Machine Learning Mania 2026 - 분석 및 모델링 기본 골자

본 문서는 캐글(Kaggle) "March Machine Learning Mania 2026" 대회의 데이터를 분석하고 예측 모델을 구축하기 위한 핵심 전략과 파이프라인 구조를 정리한 것입니다.

## 1. 대회 핵심 목표 (Objective)
*   **과제:** 과거에 한 번도 맞붙은 적 없는 팀들을 포함하여, 토너먼트에 진출 가능한 모든 팀 간의 1:1 매치업에 대해 승리 확률 예측.
*   **평가 지표 (Metric):** **Brier Score** (이 맥락에서는 **MSE, 평균 제곱 오차**와 동일).
    *   예측 확률과 실제 결과(승리 1, 패배 0) 차이의 제곱 평균을 최소화하는 것이 핵심 목표입니다.
    *   **Log Loss와의 차이점:** 과거 대회에서 사용된 Log Loss는 100% 확신하고 틀렸을 때 무한대의 페널티를 부여하므로 확률값을 자르는(Clipping, 예: 0.025 ~ 0.975) 꼼수가 필수적이었습니다. 반면 Brier Score는 최대값이 1로 제한되므로 무한대 감점이 일어나지 않아 억지로 예측값을 자르는 Clipping 과정이 필요하지 않습니다. 평가/검증 지표를 반드시 Brier Score(또는 MSE)로 설정하여 학습시켜야 합니다.

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

## 4. 데이터 분할 및 교차 검증 (Cross-Validation)
모델 학습 시 무작위 분할(Random Split)이나 계층화 K-Fold(Stratified K-Fold)는 지양하며, 미래를 과거의 데이터만으로 예측해야 하는 대회의 본질에 맞추어 **시간 기반 순차 분할 (Time-based Rolling Holdout)** 전략을 모든 학습(Step 1~4) 파이프라인에 공통으로 적용합니다.

*   **배경:** 농구 트렌드의 변화(시즌 메타)를 반영하고, 동일한 매치업 데이터의 위치를 바꾼 증강 데이터(Augmentation Swap)로 인한 정보 누수(Data Leakage)를 완벽히 차단하기 위함입니다.
*   **적용 예시 (최근 5년을 Validation Target으로 삼을 경우):**
    *   **Fold 1 검증:** (2003~2020년 데이터로 학습) 👉 **2021년 토너먼트만 예측 및 채점**
    *   **Fold 2 검증:** (2003~2021년 데이터로 학습) 👉 **2022년 토너먼트만 예측 및 채점**
    *   **Fold 3 검증:** (2003~2022년 데이터로 학습) 👉 **2023년 토너먼트만 예측 및 채점**
    *   **Fold 4 검증:** (2003~2023년 데이터로 학습) 👉 **2024년 토너먼트만 예측 및 채점**
    *   **Fold 5 검증:** (2003~2024년 데이터로 학습) 👉 **2025년 토너먼트만 예측 및 채점**
*   **최종 평가:** OOF(Out-Of-Fold) 개념으로, 위 5개 Fold에서 산출된 5개의 Validation Brier Score의 평균값을 모델의 진짜 실력으로 판단하고 하이퍼파라미터 체택 여부 및 앙상블 가중치를 결정합니다.

---

## 5. 학습 전략 (Training Strategy): 점진적 모델 빌딩 (Progressive Modeling)
단일 모델 완성부터 다중 앙상블, 그리고 신경망(MLP) 결합까지 단계별로 실험하여 어떤 데이터 조합이 가장 높은 성능(Brier Score 최소화)을 내는지 검증하는 **점진적 파이프라인(Progressive Pipeline)**을 구축합니다.

### Step 1. 단일 베이스라인 모델 (Single Baseline Model)
*   **목적:** 전처리된 데이터(`base_*, advanced_*`)의 기본적인 유효성을 검증하고 정상적으로 학습/평가가 이루어지는지 확인합니다.
*   **선택 알고리즘:** `LightGBM` (가장 빠르고 훌륭한 기본 성능) 또는 `Logistic Regression` (해석력이 좋고 안정적임).
*   **검증 방식:** Time-based 롤링 교차 검증 (예: 2018~2022 학습 -> 2023 검증, 2019~2023 학습 -> 2024 검증)을 통해 `Brier Score(MSE)`를 측정합니다.
*   **주요 리소스:** `optuna`를 활용하여 단일 모델의 주요 하이퍼파라미터(max_depth, learning_rate 등)만 가볍게 튜닝합니다.

### Step 2. 트리 기반 3대장 앙상블 (Tree-based 3-Model Ensemble)
*   **목적:** 단일 트리의 과적합(Overfitting)을 방지하고, 각 알고리즘의 장점을 결합하여 분산을 줄입니다.
*   **선택 알고리즘:** `LightGBM` + `XGBoost` + `CatBoost`
*   **결합 방식:**
    1.  **단순 평균 (Simple Average):** 예측된 확률값 3개를 더해 3으로 나눕니다. 가장 구현이 쉽고 베이스라인으로 강력합니다.
    2.  **최적 가중치 평균 (Weighted Average):** `scipy.optimize.minimize` (주로 SLSQP 메서드)를 활용하여 Validation Set의 Brier Score를 최소화하는 각 모델별 최적의 가중치(예: LGBM 0.4, XGB 0.3, CAT 0.3)를 찾습니다.

### Step 3. 메타 모델 가세 (5-Model Ensemble & Logistic Calibration)
*   **목적:** 트리 모델들이 잡지 못하는 선형적 관계를 보완하고, 최종 확률값의 신뢰도(Calibration)를 대회 평가지표에 완벽히 맞춥니다.
*   **추가 알고리즘:** `Logistic Regression` (강력한 L1/L2 정규화 포함) 추가.
*   **자체 레이팅(Part I) 휴리스틱 추가:** 머신러닝 모델이 아닌, 수학적으로 도출된 `Elo_WinProb` (Elo 레이팅 기반 승률 예측치)를 5번째 독립적인 예측 모델처럼 취급하여 병합합니다.
*   **결합 및 보정 방식:**
    *   총 5개 결과물(LGBM, XGB, CAT, LR, Elo)을 최적 가중치로 앙상블(`blend_predictions`).
    *   **Logistic Calibration (로지스틱 보정):** 앙상블된 최종 확률 예측값이 실제 0과 1 분포에 맞게 잘 스케일링 되었는지(overconfident하지 않은지) 확인하기 위해 `scipy.special.logit` 변환 후 다시 한번 1D Logistic Regression을 통과시켜 확률값을 세밀하게 깎아냅니다 (Calibration).

### Step 4. 비선형 신경망 아키텍처 결합 (Adding MLP / Neural Nets)
*   **목적:** 데이터가 풍부해지고 파생 변수(Diff, Ratio 등)의 복잡한 비선형 교차 작용을 탐지하기 위해 딥러닝 기법을 마지막에 스택킹합니다.
*   **모델 구조 제안:**
    *   `PyTorch`를 활용한 3-Layer MLP 깊은 앙상블 (예: `march-mania.ipynb` 참조).
    *   범주형 변수(Seed Number, Conference Code 등)를 처리하기 위한 임베딩(Embedding) 레이어 + 연속형 변수를 처리하는 Dense 레이어 구조.
*   **앙상블 방식 (Stacking):**
    *   Step 3까지 만들어진 트리+선형 보정 모델들의 예측 결과치 값들 자체를 새로운 피처셋으로 삼고(Meta Features), 신경망 모델이 이를 입력받아 최종 최적화 확률을 내뿜도록 아키텍처를 구성합니다 (Deep Blending).

---

## 6. 남녀 모델 분리 학습 (Gender Separation) 시스템
*   제출 양식에는 남성과 여성 팀이 서로 맞붙는 혼합 경기(Mixed Matchup)가 존재하지 않습니다.
*   따라서 남성(Men's) 데이터와 여성(Women's) 데이터를 **완전히 분리하여 두 개의 독립된 파이프라인(Step 1 ~ 4)으로 학습**시킵니다.
    *   여성 데이터는 상대적으로 이변(Upset)이 적고 상위 시드가 승리할 확률이 더 높게 나타나는 패턴이 있으므로, 하이퍼파라미터와 앙상블 가중치를 남성 모델과 별도로 도출해야 합니다.
*   추론 후, `[M_Pred_DF, W_Pred_DF]`를 마지막 `submission.csv` 작성(Concat) 시점에만 하나로 이어 붙입니다.

---

## 7. 최종 예측 및 제출 (Prediction & Submission)
2026년 대회가 실제로 개최되어 대진 예측을 할 때의 파이프라인입니다.

1.  2026년 정규 시즌 성적을 바탕으로 완성된 [Step 1~4] 파이프라인 전처리 코드를 가동합니다.
2.  `SampleSubmissionStage1.csv` (또는 Stage2)에 적힌 모든 가상 매치업(`ID: 2026_1101_1102`)을 읽어옵니다.
3.  성별 식별자에 맞춰 남성 모델 / 여성 모델에 각각 통과시켜 승리 예측 확률(Pred)을 도출합니다.
4.  지정된 형식(`ID`, `Pred`)에 맞춰 `submission.csv` 파일을 생성합니다. 무한대 감점을 막기 위해 Brier Score 최적화가 안 된 상태라면 예측값을 절대 0이나 1로 두지 않고 (예: `clip(0.001, 0.999)`) 안전 장치를 걸어둡니다.
