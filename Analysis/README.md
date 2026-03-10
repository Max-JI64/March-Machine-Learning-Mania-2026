# March Machine Learning Mania 2026 - 분석 및 모델링 기본 골자

본 문서는 캐글(Kaggle) "March Machine Learning Mania 2026" 대회의 데이터를 분석하고 예측 모델을 구축하기 위한 핵심 전략과 파이프라인 구조를 정리한 것입니다.

## 0. 구동 환경
runpod 환경에서 구동할때는 아래의 라이브버리 버전을 설치하세요.  
캐글과 동일한 버전의 라이브러리입니다.
```bash
pip install pandas==2.3.3 numpy==2.0.2 scikit-learn==1.6.1 optuna==4.7.0 xgboost==3.1.3 lightgbm==4.6.0
```

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

### Step 3. 메타 모델 가세 (5-Model Ensemble & Calibration)
*   **목적:** 트리 모델들이 잡지 못하는 선형적 관계를 보완하고, 최종 확률값의 신뢰도(Calibration)를 대회 평가지표에 완벽히 맞춥니다.
*   **추가 알고리즘:** `Logistic Regression` (강력한 L1/L2 정규화 포함) 추가.
*   **자체 레이팅(Part I) 휴리스틱 추가:** 머신러닝 모델이 아닌, 수학적으로 도출된 `Elo_WinProb` (Elo 레이팅 기반 승률 예측치)를 5번째 독립적인 예측 모델처럼 취급하여 병합합니다.
*   **결합 및 보정 방식:**
    *   총 5개 결과물(LGBM, XGB, CAT, LR, Elo)을 최적 가중치로 앙상블(`blend_predictions`).
    *   **Isotonic Calibration / Logistic Calibration:** 앙상블된 최종 확률 예측값이 실제 0과 1 분포에 맞게 잘 스케일링 되었는지 확인하고, `IsotonicRegression` 또는 `LogisticRegression`을 통해 확률값을 세밀하게 보정합니다. (Brier Score 개선 핵심)

### Step 4. 비선형 신경망 아키텍처 및 고급 앙상블 (NN & MoE)
*   **목적:** 데이터가 풍부해지고 파생 변수의 복잡한 비선형 교차 작용을 탐지하기 위해 딥러닝 및 고급 앙상블(Mixture of Experts) 기법을 활용합니다.
*   **모델 구조 제안:**
    *   `PyTorch`를 활용한 3-Layer MLP (128→64→32) + Dropout(0.3) + Label Smoothing(0.05).
    *   **MoE 스타일 융합:** FFM(Factorization Machine) + GBDT + LR + NN의 예측값들을 Meta-Feature로 활용하여 다시 신경망이나 XGBoost로 학습시키는 이중 경로 융합 (Path-based Fusion).
*   **데이터 증강 (Advanced Augmentation):** 가우시안 노이즈(`N(0, 0.02)`) 추가 및 30% 샘플의 피처/라벨 반전(Flip) 등을 통해 모델의 일반화 성능을 극대화합니다.

---

## 6. 남녀 모델 분리 학습 (Gender Separation) 시스템
*   제출 양식에는 남성과 여성 팀이 서로 맞붙는 혼합 경기(Mixed Matchup)가 존재하지 않습니다.
*   따라서 남성(Men's) 데이터와 여성(Women's) 데이터를 **완전히 분리하여 두 개의 독립된 파이프라인(Step 1 ~ 4)으로 학습**시킵니다.
    *   **성별 특성 차이:** 남성은 업셋(Upset) 비율이 높고(약 27%), 여성은 상위 시드가 승리할 확률이 더 높은 경향(약 21%)이 있습니다. 이를 반영하여 하이퍼파라미터를 각기 다르게 튜닝해야 합니다.
*   추론 후, `[M_Pred_DF, W_Pred_DF]`를 마지막 `submission.csv` 작성 시점에만 하나로 이어 붙입니다.

---

## 7. 타인의 분석 사례 참고 (Reference Analysis Cases)
`Example/Model_Analysis.md`에 정리된 3가지 주요 접근 방식을 단계별 실험에 참고합니다.

### Case 1. 정교한 피처 엔지니어링 & Brier 가중 앙상블
- **핵심:** 99개 피처 (Elo, Massey, Four Factors, Momentum 등).
- **앙상블:** LGB + XGB + CAT + LR 4모델 앙상블.
- **특이점:** **Inverse-Brier 가중 평균** (보정 성능이 좋은 모델에 높은 가중치) + **Isotonic Calibration** 적용.

### Case 2. 심플한 피처 & 스택킹 메타 모델
- **핵심:** 16개 핵심 피처 (시드 차이, 평균 득실 차이 등) 위주.
- **앙상블:** LGB + XGB + CAT 5-fold CV 예측값을 입력으로 받는 **Logistic Regression Stacking**.
- **특이점:** 복잡한 피처보다 모델 간의 앙상블(Stacking) 효과에 집중.

### Case 3. 대규모 피처 & 3-Layer MoE 아키텍처
- **핵심:** 188개 피처 (고급 팀 통계, SOS, Recency 등).
- **구조:** 
    - Layer 1: FFM, GBDT, LR, NN (총 11개 베이스 모델).
    - Layer 2: MoE 스타일 이중 경로 융합 (은닉 요인 융합 + Pctr 점수 융합).
    - Layer 3: 최종 메타 모델 학습.
- **특이점:** FFM(Factorization Machine)을 통한 피처 교호작용 탐지 및 복잡한 신경망 융합.

---

## 8. 최종 예측 및 제출 (Prediction & Submission)
2026년 대회가 실제로 개최되어 대진 예측을 할 때의 파이프라인입니다.

1.  2026년 정규 시즌 성적을 바탕으로 완성된 [Step 1~4] 파이프라인 전처리 코드를 가동합니다.
2.  `SampleSubmissionStage1.csv` (또는 Stage2)에 적힌 모든 가상 매치업을 읽어옵니다.
3.  성별 식별자에 맞춰 남성 모델 / 여성 모델에 각각 통과시켜 승리 예측 확률(Pred)을 도출합니다.
4.  **안전 장치:** Brier Score 최적화가 안 된 상태라면 예측값을 절대 0이나 1로 두지 않고 `clip(0.025, 0.975)` 등으로 안전 범위를 설정합니다.

---

## 9. 모델 훈련 시 모니터링 토큰/비용 최적화 팁 (Token Optimization)
모델 학습(특히 Optuna 튜닝이나 K-Fold 교차검증 등)은 긴 시간이 소요되므로, 터미널 로그를 실시간으로 계속 확인하려고 하면 대기 시간과 불필요한 토큰 소모가 크게 발생합니다. 이를 방지하고 효율적으로 작업하기 위한 권장 워크플로우는 다음과 같습니다:

1. **학습 실시간 모니터링 및 작업 끊기**: 코드를 작성하고 실행할 때, 명령어 출력을 터미널과 로그 파일에 동시에 실시간으로 출력하도록 `tee` 명령어를 사용합니다. (예: `python -u script.py 2>&1 | tee train.log`) 그리고 실행 명령을 내린 직후 AI 코딩 에이전트의 작업을 완전히 중단(종료)시켜 불필요한 로그 분석 토큰 소모를 방지합니다.
2. **트리 파라미터 Verbose 최소화**: XGBoost, LightGBM, CatBoost 모델 학습 시 `verbose=False` 나 `verbose=-1` 옵션을 부여하여 저장되는 로그 파일의 크기가 과도하게 커지지 않도록 합니다.
3. **학습 완료 후 결과 분석 지시**: 학습이 완전히 종료된 후(사용자가 직접 종료 여부 확인), AI에게 **"이제 로그(train.log)를 확인하고, 다음으로 개선할 수 있는 방안을 탐색하고, 코드를 만들고 실행해"** 라고 새로운 프롬프트를 입력합니다.
4. **이전 작업 로그 확인**: AI는 새로운 프롬프트를 받고 나면, 이전 작업에서 생성되어 저장된 `train.log` 파일을 읽어들여 전체 학습 결과를 한 번에 분석하고 다음 스텝을 지연 없이 진행할 수 있습니다.
