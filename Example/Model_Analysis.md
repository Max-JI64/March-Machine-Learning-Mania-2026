# March Mania 2026 Example Notebooks 분석

`/Users/swoo64/Desktop/March-Machine-Learning-Mania-2026/Example` 디렉토리에 있는 6개의 예시 코드 파일들에 대한 데이터 사용, 전처리 과정, 그리고 모델 학습 및 예측 방식을 분석한 내용입니다.

---

## 1. elo-massey-ordinals-four-factors-ensemble.ipynb
**주요 특징:** 강력한 Elo 레이팅(마진/홍구장 보정) 및 Massey Ordinals 기반 앙상블 모델

- **사용 데이터:** `*CompactResults`, `*DetailedResults`, `*Seeds`, `*TeamConferences`, `MMasseyOrdinals`, `MTeamCoaches`
- **전처리 및 피처 엔지니어링 (Feature Engineering):**
  - **Elo Rating:** 점수 차이(마진)에 따른 K-factor 조정 (`k_regular=20`, `k_tourney=30`). 토너먼트 경기에 가중치를 두며, 승리 마진에 로그 변환된 가중치(Margin Multiplier)를 적용: `np.log(margin + 1) / np.log(cap + 1)`. 홈 코트 어드밴티지 보정 (`home_bonus=75`를 기대 승률 공식에 반영). 시즌 시작시 기본값(`1500`)에 기반해 시즌 간 평균 회귀(`reversion=0.30`)를 적용.
  - **Massey Ordinals (남성):** 대회 전(DayNum <= 133)을 기준으로 7개 시스템(POM, SAG, COL, DOL, MOR, WLK, RTH) 랭킹을 가져옴. 팀 수가 다름을 보정하기 위해 각 시스템 순위를 백분위 수로 변환 (`(x.max() - x + 1) / x.count()`). 시스템별 평균, 최솟값, 표준 편차 계산.
  - **Four Factors (딘 올리버 4요소):** `DetailedResults` (DayNum <= 132 정규시즌 한정)를 처리해 4요소를 산출:
    - `eFG% = (FGM + 0.5*FGM3) / FGA`
    - `TOVPct = TO / (FGA + 0.44*FTA + TO)`
    - `ORBPct = OR / (OR + Opp_DR)`
    - `FTRate = FTM / FGA`
    이를 기반으로 부가적으로 Margin의 표준편차(MarginStd) 및 자유투 비율 표준편차(FTRateStd) 등을 생성.
  - **기타 보조 피처:** 
    - **SOS (Strength of Schedule):** 정규시즌 상대팀들의 Pre-tournament Elo 평균 (`OppElo`).
    - **Momentum:** 최근 10경기 승률에 지수 가중치 최신순으로 점근 (`decay=0.85` 제곱을 뒤에서부터 적용하여 스코어화) `Weight = 0.85 ** RevIdx`.
    - **H2H (상대전적):** 베이지안 스무딩을 통한 역사적 상대전적 승률 도출 `H2H = (Wins + 3*0.5)/(Games + 3)`.
    - **Coach Records:** 감독의 커리어 토너먼트 승률 (마찬가지로 스무딩 처리, 처음 나온 감독은 0.5로 대체).
- **학습 및 예측:**
  - **모델:** LightGBM, XGBoost, CatBoost. 각 모델 분류기(Classifier) 객체를 만들어 시간(Time-based) 기반 분할을 사용 (2008~2022년 Train, 2023~2025년 Validation). 베이지안 파라미터 최적화 라이브러리인 Optuna의 TPESampler를 이용해 TPE 방식으로 하이퍼파라미터를 최적화.
  - **보정(Calibration) 및 블렌딩:** 3개 모델 결과 확률을 결합 시에 Isotonic Regression으로 확률 캘리브레이션을 진행. 최종 확률은 `0.025`에서 `0.975` 값으로 클리핑(비관적 극단값 보정). 앙상블 조합 최적화에 SciPy의 `minimize` 기능을 활용해 Brier Score Loss가 최소가 되는 가중치(weights)를 구하여 결합함.

---

## 2. lightgbm-xgboost-ensemble.ipynb
**주요 특징:** 단순 통계 피처 및 메타 모델(Stacking Meta-model) 앙상블

- **사용 데이터:** `*CompactResults`, `*Seeds`
- **전처리 및 피처 엔지니어링:**
  - **Seed Feature:** 시드 번호 문자열을 정수형 번호(`Seed_Num`)로 변환 (`df['Seed'].str[1:3].astype(int)`). 그 외 시드 파생 변수 생성:
    - 강도: `Seed_Strength = 17 - Seed_Num`
    - 역수 가치: `Seed_Value = 1 / Seed_Num`
    - 제곱: `Seed_Squared = Seed_Num ** 2`
    - 백분위: `Seed_Percentile = (17 - Seed_Num) / 16`
  - **기본 팀 통계:** 득점, 실점, 마진, 승률의 시즌별/팀별 산출 (`Compact` 기반). `Avg_pts_scored`, `Avg_pts_allowed`, `Avg_margin`, `Win_pct`
  - **매치업 피처:** 팀1(승자)과 팀2(패자) 간 시드 번호 차이(`Seed_Num_Diff`), 비율(`Seed_Num_Ratio`), 팀 스탯 승률 차이(`Win_pct_Diff`) 생성.
  - **시드 티어 구분 지표:** 4시드마다 등급을 메겨 One-Hot 플래그화. (1~4는 Elite, 5~8은 Contender, 9~12는 Mid, 13~16은 Low). 같은 티어 여부를 확인하는 추가 지표 (`Same_Tier_Elite`, `Same_Tier_Low`).
- **학습 및 예측:**
  - **학습 데이터 구성:** 모든 승리팀 row(`Win=1`) 데이터에 패배팀 row(`Win=0`) 데이터를 위치를 바꾸어 추가 결합하여 모델 학습용 이진 분류 데이터 생성. (Data augmentation)
  - **교차 검증 방식:** 5 K-Fold Stratified Cross Validation (`StratifiedKFold(n_splits=5)`). 
  - **속도 모드 지정 파라미터 튜닝:** 지정된 `SPEED_MODE` ("fast", "balanced", "accurate") 에 기반하여 Optuna로 Bayesian Parameter Optimization 횟수를 지정 (기본 `balanced` = 20회).
  - **모델:** LightGBM, XGBoost, CatBoost. 각 모델 Fold별 예측값을 OOF(Out-of-Fold) 예측 결과로 결합. 
  - **예측 앙상블:** 세 트리 기반 모델의 OOF 확률을(`meta_X = np.column_stack([oof_lgb, oof_xgb, oof_cb])`) 새로운 인풋 스태킹(Stacking) 데이터로 활용하여 Logistic Regression 메타 앙상블 모델 학습. 최종 메타 모델 확률 결과는 제한영역 `[0.01, 0.99]` 로 클리핑 처리.

---

## 3. march-mania.ipynb
**주요 특징:** Fast Factorization Machine (FFM) 모델을 포함한 3-layer 딥 블렌딩 기법과 Data Augmentation

- **사용 데이터:** `*DetailedResults` (스탯 기반), `*Seeds`, `MMasseyOrdinals`, `MTeamCoaches`, `*TeamConferences`
- **전처리 및 피처 엔지니어링:**
  - **Elo Rating:** 이전 경기들의 Elo 점수에 승수 마진 모멘텀을 가중치로 둔 업데이트: `d = k * mov * (1 - ew)`. (초기값 1500, K=20).
  - **고급 리그스탯 산출 (Advanced Stats):** `compute_team_stats` 함수로 전경기 박스스코어 누적 특징 산출. 주요 산출식:
    - Possession: `Poss = FGA - OR + TO + 0.44*FTA`
    - 유효 슈팅 비율 (eFG%): `(FGM + 0.5*FGM3) / FGA`
    - 3점 시도 비율 (3PAR): `FGA3 / FGA`
    - 오펜시브/디펜시브 레이팅 (OffRtg/DefRtg): `Score / Poss * 100` 및 `OppScore / Poss * 100`
    - 피타고리안 승률 기댓값 (Pyth): `Score**11.5 / (Score**11.5 + OppScore**11.5)`
    이 외에 마지막 N(14) 경기(Recency)에 대한 평균 스탯을 별도로 산출.
  - **SOS (Strength of Schedule):** 상대의 평균 방어 레이팅 (OppDefRtg).
  - **매치업 인터랙션 특징:** 팀 간의 통계값 차이 `T1_c - T2_c` 생성 외에도, 곱연산된 인터랙션 스코어 생성:
    - `Elo_x_SeedDiff = EloDiff * SeedDiff`
    - `UpsetScore = np.abs(SeedDiff) * (1 - np.abs(EloDiff) / 200)`
    - `HotnessScore = np.abs(SeedDiff) * D_Win_mean` (근래 14경기 승률 기준 Upset Potential)
- **학습 및 예측:**
  - **앙상블 구조 (Three-Layer Stacking/Blending):**
    - **Layer 1 Base Models:** 
      - 고유 구현한 Fast Factorization Machine (FFM) 모델 3개 (Random 노이즈 스플릿 추가 학습). FFM은 Batch 기울기 하강법과 임베딩 상호작용 반영식 도출 `1 / (1 + np.exp(-score - 0.1 * interactions))`을 사용.
      - 기본 분류기: LightGBM 2개, XGBoost 1개, CatBoost 2개, 서로 다른 시드를 가진 Logistic Regression(RidgeClassifier) 3개.
      - Neural Network (NN): PyTorch로 구성된 5겹 은닉층의 다층 퍼셉트론 (Layer 2개짜리 병렬 Network 임베딩 결합).
    - **가우시안 노이즈 증강 (Data Augmentation):**
      - 트리 기반 모델과 신경망의 Robustness(강건성)을 위해 각 트레이닝 셋에 표준 편차 0.05 크기의 가우시안 노이즈를 더해 데이터 볼륨을 2배로 증가 (`X_aug = X + np.random.normal(0, np.std(X, axis=0) * noise_level, X.shape)`). 
    - **Layer 2 & Layer 3 메타 예측:** 모든 Layer 1 결과를 OOF(Out-of-Fold) 예측으로 모아 다시 Optuna를 통해 선형 계수 비율 가중치를 학습하여 최종 결합. 메타 비율을 각각 구하고 최종적으로 세 모델을 Average Blending 함.

---

## 4. march-ml-mania-2026-lgbm-xgb-catboost.ipynb
**주요 특징:** 가벼운 데이터셋 기반의 단순 부스팅 앙상블 평균(Voting)

- **사용 데이터:** `*CompactResults`, `*Seeds`, `MMasseyOrdinals`
- **전처리 및 피처 엔지니어링:**
  - 극히 단순화된 피처 군 구성. (정규시즌 통계 산출 `build_season_stats`).
  - **정규시즌 통계:** 득점(`ScoreFor = WScore/LScore`), 실점(`ScoreAgainst`), 승수(`Wins`), 총 경기수(`Games`), 마진(`Margin = ScoreFor - ScoreAgainst`), 승률(`WinRate = Wins / Games`)의 팀별/시즌별 누적. 
  - **시드 정보:** 시드의 숫자 데이터만 파싱하여 병합 (`int(''.join(filter(str.isdigit, s)))`).
  - **Massey Ordinals:** `SystemName == 'MOR'`인 랭킹만을 선택하여 가장 최신(마지막) `OrdinalRank`를 추출 (`.groupby(['Season','TeamID']).last()`).
  - **매치업 피처 구축:** T1(어느 팀인지 무관하게 작은 TeamID로 통일)과 T2간 대결 승부에서 양측의 Seed, WinRate, Margin, Score 평점, MOR 순위를 합친 뒤 차이 피처 생성.
    - `SeedDiff = T1_Seed - T2_Seed`
    - `WinRateDiff = T1_WinRate - T2_WinRate`
    - `MarginDiff = T1_Margin - T2_Margin`
    - `ScoreDiff = T1_Score - T2_Score`
    - `MORDiff = T1_MOR - T2_MOR`
- **학습 및 예측:**
  - **학습 데이터 구성:** 결측치 처리 (`global_med = df.median().fillna(0)`) 후 Stratified 5-Fold KFold 분할 (`StratifiedKFold(n_splits=5, shuffle=True)`). 남/녀 모델 분리 학습. 시드 번호는 카테고리형(Categorical) 변수로 지정.
  - **학습 모델 설정값 파라미터 고정:** 최적화를 외부에서 수행하거나 경험적으로 도출한 고정 파라미터 활용 (예: LightGBM `learning_rate=0.05, num_leaves=31`, XGBoost `max_depth=5, tree_method='hist'`).
  - **예측:** 최종 예측 스코어는 3개 모델(LightGBM, XGB, CatBoost)이 예측한 OOF 및 Test Probabilities 를 단순 평균(`(prob_lgb + prob_xgb + prob_cat) / 3`)으로 투표(Voting) 결합한 뒤, 제한영역 클리핑 (`sub_final['Pred'].clip(0.025, 0.975)`) 적용.

---

## 5. ncaa-2026-eda-elo-ratings-and-gradient-esemble.ipynb
**주요 특징:** EDA 중심의 접근과 Brier Score 기반 트리 평가지표 결합 평균 모델

- **사용 데이터:** `*CompactResults`, `*DetailedResults`, `*SecondaryTourneyCompactResults`, `*Seeds`, `*TeamConferences`, `MTeamCoaches`
- **전처리 및 피처 엔지니어링:**
  - **Elo Rating:** 정규시즌 외에 투어리먼트 및 2차 대회 경기까지 모두(SecondaryTourney) 포괄하여 Elo 점수 계산. 홈 코트 어드밴티지 보상을 직접 계산에 포함 (`HOME=100`) 점수 보상 방식 사용 (`ha = HOME if wl == "H" else (-HOME ...)`). 시즌 계승 비율은 `REV=0.75`로 세팅.
  - **상세 통계 지표 (compute_stats):** 20개 이상의 매우 다양한 세밀한 지표를 산출.
    - 게임 당 공격 스탯: 득점(`oeff = pts / poss * 100`), eFG% (`efg = (fgm + 0.5 * f3m) / fga`), 자유투 비율(`ftr = ftm / fga`), 3점 성공률(`f3pct`).
    - 방어 및 리바운드 스탯: 방어율(`deff`, `oorpct`, `oftr`), 리바운드 확률(`orpct = orb / (orb + odrb)`), 고도화된 턴오버율(`tor = to / poss`).
    - 스피드 지표: 페이스 (`pace = (poss + oposs) / (2 * n)`).
  - **SOS 기반 추가 지표:** 기존 단순 Elo 상대 평균 외에도, 승리한 상대들의 평균 Elo 점수를 고려한 Quality Wins 피처들을 구축.
- **학습 및 예측:**
  - **학습 데이터 구성:** 팀1, 팀2 스탯 갭(`T1_wpct - T2_wpct` 등)과 시드갭에 더해 Elo 차이 등을 활용.
  - **모델:** LightGBM, XGBoost, CatBoost. 각 모델의 성능 평가지표로 Brier Score (평균 제곱 오차 확률판별식) 을 측정(`brier_score_loss`).
  - **예측 앙상블:** 모델 파라미터 최적화 없이 단순 Voting 형태의 앙상블 취합을 진행하나, 내부적으로 결과 확률값을 교정(Calibration)하여 확률이 좀 더 부드럽게 매핑되게끔 Brier Score 기반 분석 결과를 시각화하고 최종 결합 확률을 도출함.Brier Score(브리어 스코어) 성능을 평가 기준으로 놓고 캘리브레이션 분포도 시각화, 지표를 확인하여 하이퍼파라미터를 결정. 예측결과는 1/3씩 등가 가중을 한 평균값 사용. 

---

## 6. strong-mania-2026-forcasting-4-model-ensemble.ipynb
**주요 특징:** 남녀 통합 피처 엔진 및 다양한 외부 레이팅 지표 (Elo + SRS) 기반의 강력한 회귀식 블렌딩 (SLSQP / Logit Calibration)

- **사용 데이터:** `*DetailedResults`, `*CompactResults`, `*Seeds`, `*TeamConferences`, `MMasseyOrdinals`
- **전처리 및 피처 엔지니어링:** (남/녀 `GenderFlag`를 활용해 파이프라인 단일화)
  - **기본 스탯 (Advanced Metrics):** `build_team_game_rows` 에 의한 세부 Box Score 통계화
    - 포제션 계산: `Poss = FGA - OR + TO + 0.475 * FTA` (4요소를 변형한 상수 0.475 사용)
    - 오펜시브/디펜시브 레이팅: `OffRtg = 100 * (PF / Poss)`, `DefRtg = 100 * (PA / OppPoss)`
    - 주요 비율 스탯: `eFG` (유효슈팅), `TOVPct` (턴오버비율), `ORBPct` (공리비율), `FTR` (자유투비율) 생성.
  - **가중치 부여 및 Recency:** `RecencyW = np.exp((DayNum - 132) / 30.0)` 지수 디케이 역산으로 대회 후반의 경기에 가중치를 줌 (`_wmean` 지표들).
  - **복합 레이팅 피처 구현:**
    - **Elo:** `build_elo` 함수로 시즌계승계수(25% 회귀) 및 홈어드밴티지 보정된 클래식 Elo 부여. 그 후 Z-스코어 정규화(`EloZ`)
    - **SRS (Simple Rating System):** Point Differential Network 행렬 방정식을 풀어서 팀의 득실마진(`mov`) 과 상대전적 행렬(M) 역행렬 계산 `rating = np.linalg.solve(A, mov)`으로 구조적 강도 점수 산출 (`ridge=0.08` 정규화).
    - **SOS (Strength of Schedule):** 상대의 `WinPct` 와 `NetRtg_wmean` 에 이전 계산된 `RecencyW` 가중치를 두어 SOS 통계(SOS_WinPct / SOS_NetRtg) 생성. 이를 바탕으로 `AdjNetRtg = NetRtg_wmean - SOS_NetRtg`.
    - **Power Composite & Expected Seed:** `AdjNetRtg_z`(0.30) + `EloZ_z`(0.23) + `SRS_z`(0.20) + `WinPct_z`(0.12) + `NetRtg_recent_z`(0.10) + `Margin_recent_z`(0.05) 를 합하여 파워 지표 결합, 이를 역산해 기대 시드(Expected Seed) 부여.
  - **매치업 피처:** T1과 T2 스탯 차이 및, 위에서 만든 PowerComposite, SRS, Elo 등 스탯의 `_diff` 및 `_sum` 계산 생성. 
- **학습 및 예측:**
  - **학습:** SimpleImputer(median) 으로 결측치 채운 뒤 Standard Scaler -> (LogisticRegression, LightGBM, XGBoost, CatBoost) Base 4모델 학습. 
  - **Holdout CV:** 최근 N(10)년치의 시즌별 Rolling Holdout `(Season < val_season) -> (Season == val_season)` 방식으로 Time-series split. 예측 평가지표는 Brier 스코어 오차 기준.
  - **블렌딩 및 캘리브레이션 조정:** 
    - 4개의 트리/선형 모델 + 사전 도출된 `EloWinProbNeutral` 모델 확률 5개의 예측값을 종합.
    - SciPy의 `minimize(method='SLSQP')` 제약조건(합=1.0) 하 최소자승 최적화로 Brier Square Loss 최저가 되는 앙상블 계수(Weights) 학습. 
    - 최종 혼합 결과물(Blend Prediction)을 Logit Calibration: Logit함수를 거친 확률값을 1차원 피처로 만들어 Logistic Regression(`C=50`)을 메타 학습시켜 Extreme Probability 보정(0.02~0.98 clip).
