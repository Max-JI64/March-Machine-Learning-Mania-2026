# March Mania 2026 Example Notebooks 분석

`/Users/swoo64/Desktop/March-Machine-Learning-Mania-2026/Example` 디렉토리에 있는 6개의 예시 코드 파일들에 대한 데이터 사용, 전처리 과정, 그리고 모델 학습 및 예측 방식을 분석한 내용입니다.

---

## 1. elo-massey-ordinals-four-factors-ensemble.ipynb
**주요 특징:** Elo 레이팅(마진/홈코트 보정) + Massey Ordinals + Dean Oliver 4요소 + 다수 보조 피처 기반 4모델 앙상블 (LGB+XGB+CAT+LR), Isotonic Calibration 적용

### 1.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | Elo, SOS, Momentum, Conference 계산 |
| `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` | Four Factors 및 Box Score 스탯 (lazy loading) |
| `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` | Elo 전체 궤적, Coach 기록, H2H, 학습 라벨 |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 번호, Tournament Experience |
| `MTeamConferences.csv` / `WTeamConferences.csv` | 컨퍼런스 소속/승률 |
| `MMasseyOrdinals.csv` | 7개 외부 랭킹 시스템 (남성 전용, 5.7M rows) |
| `MTeamCoaches.csv` | 감독 토너먼트 승률 (남성 전용) |
| `MTeams.csv` / `WTeams.csv` | 팀 이름 (시각화용) |

### 1.2 전처리 및 피처 엔지니어링 (Feature Engineering)

> **총 99개 피처** = 30개 팀 피처 × 3개 뷰(T1 값, T2 값, T1−T2 차이) + 9개 매치업 전용 피처

#### A. Elo Rating (`compute_elo`) — 팀 피처: `Elo` (1개)

정규시즌+토너먼트 경기 합산(`pd.concat([reg_c, tour_c])`)으로 전체 시즌 궤적 계산.

- **초기값:** `elo_init = 1500`
- **시즌 간 평균 회귀 (Mean Reversion):**
  ```python
  ratings[tid] = ratings[tid] + reversion * (init - ratings[tid])   # reversion = 0.30
  ```
  매 시즌 시작 시 전체 팀의 레이팅을 1500 쪽으로 30% 회귀.
- **기대 승률 (Expected Score):**
  ```python
  E(A) = 1.0 / (1.0 + 10.0 ** ((r_B - r_A) / 400.0))
  ```
- **홈 코트 어드밴티지:**
  - `WLoc == 'H'` → `E = _expected(r_W + 75, r_L)` (승자가 홈)
  - `WLoc == 'A'` → `E = _expected(r_W, r_L + 75)` (패자가 홈)
  - `WLoc == 'N'` → 보정 없음 (중립)
  - **주의:** 보너스는 기대 승률 계산에만 적용, 저장된 레이팅 자체에는 반영 안 됨.
- **마진 승수 (Margin Multiplier):**
  ```python
  multiplier = ln(min(|margin|, 25) + 1) / ln(25 + 1)   # cap = 25
  ```
  FiveThirtyEight 방식의 로그 스케일 마진 보정.
- **K-factor:** 정규시즌 `k_regular=20`, 토너먼트(`DayNum≥134`) `k_tourney=30`
  ```python
  k_adj = k_base * margin_multiplier
  delta = k_adj * (1.0 - E_winner)
  ```
- **출력:** `EloPreTourney` (DayNum≥134 직전 스냅샷), `EloEndSeason` (시즌 종료 시점)
- 남/녀 각각 별도로 계산 → `m_elo_lu`, `w_elo_lu` (딕셔너리: `(Season, TeamID) → float`)

#### B. Massey Ordinals (`build_massey_features`) — 팀 피처: 5개 (남성 전용, 여성은 기본값 0.5)

- **대상 시스템:** `['POM', 'SAG', 'COL', 'DOL', 'MOR', 'WLK', 'RTH']` (총 7개)
- **필터:** `RankingDayNum <= 133` (pre-tourney), 각 시스템별 가장 최신 랭킹 (`idxmax()`)
- **백분위 변환:**
  ```python
  percentile = (x.max() - x + 1) / x.count()   # 시즌×시스템 단위, ∈ [0,1], 높을수록 좋음
  ```
- **파생 변수:**
  | 변수명 | 산출식 |
  |--------|--------|
  | `mas_POM` ~ `mas_RTH` | 각 시스템별 백분위 (7개, 이 중 `mas_POM`과 `mas_SAG`만 팀 피처로 사용) |
  | `mas_mean` → `MasMean` | 7개 시스템 백분위의 **평균** |
  | `mas_min` → `MasMin` | 7개 시스템 중 **최저 백분위** (가장 안 좋은 랭킹) |
  | `mas_std` → `MasStd` | 7개 시스템 백분위의 **표준편차** (랭킹 일관성) |
  | `mas_n_sys` | 랭킹이 존재하는 시스템 수 (사용되지 않음) |

#### C. Four Factors & Box Score (`compute_four_factors`) — 팀 피처: 14개

`DetailedResults` 로부터 (DayNum ≤ 132, 정규시즌만) 산출. Lazy loading으로 메모리 절약.

**기본 누적 집계 (시즌×팀 단위 sum):**
- `Win`, `Score`, `Allowed`, `FGM`, `FGA`, `FGM3`, `FGA3`, `FTM`, `FTA`, `OR`, `DR`, `Ast`, `TO`, `OppDR` + `Games`(=count)

**파생 변수 (총 14개):**

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Games` | `count(Win)` | 총 경기 수 |
| `WinPct` | `Win / Games` | 승률 |
| `AvgScore` | `Score / Games` | 평균 득점 |
| `AvgAllow` (`AvgAllowed`) | `Allowed / Games` | 평균 실점 |
| `AvgMargin` | `AvgScore − AvgAllow` | 평균 마진 |
| `MarginStd` | `std(Score − Allowed)` 경기별 | 마진 표준편차 (일관성 지표) |
| `FTRateStd` | `std(FTM / FGA)` 경기별 | 자유투율 변동성 |
| `eFGPct` | `(FGM + 0.5 × FGM3) / (FGA + ε)` | 유효 슈팅 % (Four Factor #1) |
| `TOVPct` | `TO / (FGA + 0.44 × FTA + TO + ε)` | 턴오버율 (Four Factor #2) |
| `ORBPct` | `OR / (OR + OppDR + ε)` | 공격 리바운드율 (Four Factor #3) |
| `FTRate` | `FTM / (FGA + ε)` | 자유투 비율 (Four Factor #4) |
| `FGPct` | `FGM / (FGA + ε)` | 필드골 성공률 |
| `FG3Pct` | `FGM3 / (FGA3 + ε)` | 3점 성공률 |
| `AstTORat` | `Ast / (TO + ε)` | 어시스트 대 턴오버 비율 |

> `ε = 1e-6` (0으로 나누기 방지)

#### D. Pythagorean Expectation — 팀 피처: `Pyth` (1개)

`get_team_features()` 함수 내에서 동적 계산 (별도 전처리 함수 없음):
```python
Pyth = AvgScore^11.5 / (AvgScore^11.5 + AvgAllow^11.5 + ε)   # 지수 = 11.5
```
Bill James / Daryl Morey의 대학 농구용 피타고리안 승률 공식. 기대 승률을 득실점 비율로부터 추정.

#### E. SOS (Strength of Schedule) (`compute_sos`) — 팀 피처: `SOS` (1개)

- **범위:** 정규시즌 (`DayNum ≤ 132`) 경기만 대상
- **산출:** 해당 시즌 해당 팀이 상대한 모든 팀의 **Pre-Tournament Elo 평균**
  ```python
  SOS = mean(OppElo)   # 상대팀 EloPreTourney 값, 미존재 시 1500으로 대체
  ```
- Winner/Loser 양쪽 관점으로 스택 → `groupby(['Season','TeamID'])['OppElo'].mean()`

#### F. Momentum (`compute_momentum`) — 팀 피처: `Momentum` (1개)

- **범위:** 정규시즌 (`DayNum ≤ 132`) 경기, **최근 10경기**만 사용
- **지수 감쇠 가중치:**
  ```python
  RevIdx = groupby(['Season','TeamID']).cumcount(ascending=False)  # 0 = 가장 최근
  Weight = 0.85 ^ RevIdx   # decay = 0.85
  Momentum = Σ(Win × Weight) / Σ(Weight)
  ```
  시즌 말 10경기에서 가중 승률을 계산, 최근 경기에 더 높은 비중.

#### G. Conference 피처 (`compute_conf_features`) — 팀 피처: `ConfWR`, `IsPower` (2개)

- **ConfWR:** 해당 시즌 소속 컨퍼런스 전체의 승률
  ```python
  WinRate = W / (W + L + ε)   # 컨퍼런스 소속 팀들의 총 승/패
  ```
- **IsPower:** 파워 컨퍼런스 소속 여부 (0 또는 1)
  - 파워 컨퍼런스 목록: `acc`, `big_ten`, `big_twelve`, `sec`, `big_east`, `pac_twelve`, `wcc`, `a_ten`

#### H. Tournament Seed (`build_seed_lu`) — 팀 피처: `Seed` (1개)

- 시드 문자열에서 숫자 추출: `Seed.str.extract(r'(\d+)').astype(int)` (예: `'W01'` → `1`)
- 시드가 없는 팀 (토너먼트 미진출)은 **기본값 `8.5`** (평균 시드)

#### I. Coach Tournament Records (`compute_coach_tourney_records`) — 팀 피처: `CoachTWR` (1개, 남성 전용)

- 감독별 역대 **토너먼트 전체** 승/패 집계 (현재 시즌 포함 — 경미한 누출 가능성)
- 토너먼트 승률:
  ```python
  CoachTWR = TWins / (TWins + TLosses + ε)
  ```
- `LastDayNum ≥ 132`인 감독 (시즌 끝까지 재임)만 대상
- 처음 나온 감독은 **기본값 `0.5`** (정보 없음)
- **여성은 `0.5` 고정** (코치 데이터 남성 전용)

#### J. H2H — Head-to-Head (`compute_h2h`) — 매치업 피처: `H2H_WinRate`, `H2H_Games` (2개)

남녀 토너먼트 합산, 정준 순서(T1 < T2)로 집계.

- **베이지안 축소 (Shrinkage):**
  ```python
  H2H_WinRate = (Wins + prior_n × 0.5) / (Games + prior_n)   # prior_n = 3
  ```
  사전 확률 50%로 3경기 분량 축소 → 소표본 과적합 방지.
- `H2H_Games` = 실제 대결 수 (최대 5로 클리핑)
- 대결 이력 없는 조합은 **기본값 `0.5`**

#### K. Tournament Experience (`compute_tourney_experience`) — 팀 피처: 4개

최근 `window=6` 시즌의 토너먼트 참가 이력 기반:

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `TourneyApps` | `count(appearances)` | 직전 6시즌 내 토너먼트 출전 횟수 |
| `TourneyWPG` | `mean(Wins per appearance)` | 출전 당 평균 토너먼트 승수 |
| `DeepRunRate` | `mean(Wins ≥ 2)` | Sweet 16+ 도달 비율 |
| `SeedAvg` | `mean(SeedNum)` | 직전 6시즌 평균 시드 (미출전 시 8.5) |

> 남녀 토너먼트 합산으로 계산.

#### L. 매치업 피처 조립 (`build_matchup_features`)

**팀 피처 (30개)를 3개 뷰로 확장:**
```python
for k in TEAM_FEAT_NAMES:   # 30개
    T1_{k} = f1[k]     # Team 1 값
    T2_{k} = f2[k]     # Team 2 값
    D_{k}  = f1[k] - f2[k]   # 차이 (differential)
```

**추가 Interaction / 확률 피처 (9개):**

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `H2H_WinRate` | Section J 참조 | 역대 상대전적 T1 승률 |
| `H2H_Games` | Section J 참조 | 역대 상대전적 경기 수 |
| `IX_Elo_x_Seed` | `D_Elo × D_Seed` | Elo-시드 교호작용 |
| `IX_Net_x_Seed` | `D_AvgMargin × D_Seed` | 마진-시드 교호작용 |
| `IX_Elo_x_Net` | `D_Elo × D_AvgMargin` | Elo-마진 교호작용 |
| `IX_Seed_x_Pyth` | `D_Seed × D_Pyth` | 시드-피타고리안 교호작용 |
| `IX_Off_x_Def` | `D_eFGPct × (−D_TOVPct)` | 공격 효율 × 수비 효율 교호작용 |
| `EloWinProb` | `1 / (1 + 10^(−D_Elo / 400))` | Elo 기반 순수 승률 |
| `SeedWinProb` | `1 / (1 + exp(0.4 × D_Seed))` | 시드 기반 시그모이드 승률 |

> **전체 피처 목록 (99개):**
> T1/T2/D × {Elo, WinPct, AvgScore, AvgAllowed, AvgMargin, MarginStd, eFGPct, TOVPct, ORBPct, FTRate, FTRateStd, FGPct, FG3Pct, AstTORat, Pyth, SOS, Momentum, Seed, ConfWR, IsPower, CoachTWR, TourneyApps, TourneyWPG, DeepRunRate, SeedAvg, MasMean, MasMin, MasStd, MasPOM, MasSAG} = 90개 + H2H_WinRate, H2H_Games, IX_Elo_x_Seed, IX_Net_x_Seed, IX_Elo_x_Net, IX_Seed_x_Pyth, IX_Off_x_Def, EloWinProb, SeedWinProb = 9개 → **총 99개**

### 1.3 학습 데이터 구성

#### 학습 레이블 설계
- **소스:** `MNCAATourneyCompactResults` + `WNCAATourneyCompactResults` (2008시즌 이후)
- **라벨 규칙:** 정준 순서 기준 (낮은 TeamID = T1), T1 승리 시 `Outcome=1`, 패배 시 `Outcome=0`
- **남녀 결합:** `IsWomen` 플래그 (0/1) 부여 후 결합, 단 **학습은 성별 분리**

#### Symmetric Augmentation (대칭 증강)
```python
# T1_* ↔ T2_* 교환, D_* 부호 반전, Outcome 반전
train_aug[t1_cols] = train_base[t2_cols].values
train_aug[t2_cols] = train_base[t1_cols].values
train_aug[d_cols]  = -train_base[d_cols].values
train_aug['Outcome'] = 1 - train_base['Outcome']
```
- 학습 데이터 **2×** 증강 → 최종 **4,432행** (base 2,216)
- 목적: TeamID 크기에 따른 편향 제거, Class balance 정확히 0.50 보장

#### Train/Validation Split
- **시간 기반 분할:** `val_seasons = [2023, 2024, 2025]` (3시즌 holdout)
- Train: 2008~2022 시즌 (증강 후)
- Validation: 2023~2025 시즌 (증강 후)
- Men: train=1,856 / val=402
- Women: train=1,772 / val=402

#### Recency Sample Weights (시즌 기반 지수 감쇠 가중치)
```python
w_tr = 0.60 ^ (max_season - Season)   # decay = 0.60
w_tr = w_tr / w_tr.mean()   # 평균 1.0 정규화
```
- 예시: 2025→1.000, 2024→0.600, 2023→0.360, 2020→0.078, 2015→0.008
- Validation 행은 항상 weight=1.0

### 1.4 피처 선택 (Shadow Feature Selection, Boruta 스타일)

**Leakage-safe 전략:** Validation 시즌을 절대 보지 않는 초기 시즌 데이터만 사용.

1. **Safe 데이터 구성:** Train 중 val_seasons 제외 → 추가로 마지막 `held_out=5` 시즌 제외
2. **Stage 1 - Probe 중요도:** LGBMClassifier (300 estimators, 31 leaves, depth=5) 학습 → Top 60 피처 추출
3. **Stage 2 - Shadow Permutation Test:** 5회 반복
   - 각 피처의 셔플된(shadow) 복사본 생성 → 병합 학습
   - `real_importance > max(shadow_importance)` 비율 계산
4. **Stage 3 - 필터링:** 셔플 대비 `threshold=0.60` 이상인 피처만 유지
5. **Stage 4 - 필수 피처 강제 포함:**
   ```python
   must_keep = ['D_Elo', 'D_Seed', 'D_MasMean', 'D_MasPOM', 'D_WinPct',
                'D_AvgMargin', 'D_eFGPct', 'D_Pyth', 'H2H_WinRate',
                'EloWinProb', 'SeedWinProb', 'D_TourneyApps', 'D_DeepRunRate']
   ```
- **결과:** 99개 → 약 **18개** 피처 선택 (남/녀 별도 수행)

### 1.5 모델 학습 및 앙상블

#### 남/녀 분리 학습 이유
- Men's upset rate: 27.3% vs Women's: 21.1%
- 별도 모델로 각 성별 토너먼트 구조 (시드 차이 중요도, 이변 확률)에 최적화

#### Optuna TPE 하이퍼파라미터 최적화

각 모델별 별도 Optuna Study (모두 `direction='minimize'`, Brier Score 기준):

**LightGBM (15 trials):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `learning_rate` | 0.01 ~ 0.10 (log) |
| `num_leaves` | 8 ~ 64 |
| `max_depth` | 3 ~ 7 |
| `min_child_samples` | 5 ~ 50 |
| `feature_fraction` | 0.5 ~ 1.0 |
| `bagging_fraction` | 0.5 ~ 1.0 |
| `reg_alpha` | 1e-3 ~ 5.0 (log) |
| `reg_lambda` | 1e-3 ~ 10.0 (log) |

**XGBoost (12 trials):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `learning_rate` | 0.01 ~ 0.10 (log) |
| `max_depth` | 3 ~ 7 |
| `subsample` | 0.5 ~ 1.0 |
| `colsample_bytree` | 0.5 ~ 1.0 |
| `min_child_weight` | 1 ~ 15 |
| `gamma` | 0.0 ~ 3.0 |
| `reg_alpha` | 1e-3 ~ 5.0 (log) |
| `reg_lambda` | 1e-3 ~ 10.0 (log) |

**CatBoost (25 trials):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `learning_rate` | 0.01 ~ 0.10 (log) |
| `depth` | 3 ~ 7 |
| `l2_leaf_reg` | 1.0 ~ 20.0 (log) |
| `bagging_temperature` | 0.0 ~ 2.0 |
| `random_strength` | 0.0 ~ 2.0 |

- **CV 전략 (Optuna 내부):** Temporal Leave-One-Season-Out — 최근 6시즌만 대상, 각 시즌을 1번씩 validation으로 사용
- **최종 학습:** Optuna 최적 파라미터로 EarlyStopping (patience=75 for LGB/XGB, 50 for CatBoost) 적용하여 재학습
  - LGB: `num_boost_round=3000`, early stop 75
  - XGB: `num_boost_round=3000`, early stop 75 (verbose=1)
  - CatBoost: `iterations=600`, early stop 50
- **Full Retrain:** 최적 iteration 수로 Train+Validation 전체 데이터(`X_all`)에 재학습

#### Logistic Regression (4번째 앙상블 멤버)
```python
SimpleImputer(strategy='median') → StandardScaler() → LogisticRegression(C=0.5, penalty='l2', solver='lbfgs', max_iter=2000)
```
- 선형 결정 경계의 다양성 추가. 스케일링 필수 (LR은 스케일에 민감).

#### 앙상블 결합 (Inverse-Brier Weighted Average)
```python
weight_i = (1.0 / Brier_i) / Σ(1.0 / Brier_j)   # 각 모델 i의 가중치
ens_pred = w_lgb × p_lgb + w_xgb × p_xgb + w_cat × p_cat + w_lr × p_lr
```
- 보정 성능(Brier Score)이 낮은 모델에 더 높은 가중치 부여.
- 실제 결과 예시: LGB=0.18, XGB=0.17, CAT=0.53, LR=0.12 (Men's)

#### Isotonic Calibration (확률 보정)
```python
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(ens_val_predictions, y_val)
calibrated = iso.predict(ens_predictions)
```
- Validation 셋의 앙상블 예측값 vs 실제 결과로 단조 변환 학습
- Brier Score 개선: 예시 Men's 0.05791 → 0.05094 (Δ = −0.00697)

### 1.6 예측 및 제출

- **성별 분기:** `t1_id >= 3000`이면 여성, 아니면 남성 → 해당 성별 모델 사용
- **각 모델 예측:** LGB, XGB, CatBoost, LogReg → 가중 평균 → Isotonic Calibration 적용
- **클리핑:** `np.clip(preds, 0.025, 0.975)` — 극단 확률값 방지
- **Stage-1:** 2022~2025 시즌 매치업 (519,144행)
- **Stage-2:** 2026 시즌 매치업 (132,133행)
- **Fallback:** 학습 실패 시 순수 Elo 기반 (`1/(1+10^((e2-e1)/400))`) 예측으로 대체

### 1.7 성능 요약

| 지표 | Men's | Women's |
|------|-------|---------|
| LGB Brier | 0.10905 | 0.11561 |
| XGB Brier | 0.11386 | — |
| CAT Brier | 0.03649 | — |
| LR Brier | 0.16494 | — |
| Ensemble Brier | 0.05791 | — |
| **Calibrated Brier** | **0.05094** | — |

---

## 2. lightgbm-xgboost-ensemble.ipynb
**주요 특징:** 시드 + 기본 팀 통계 기반 16개 피처, Optuna 튜닝 → LGB+XGB+CatBoost 5-fold CV → Logistic Regression Stacking Meta-model

### 2.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | 팀 시즌 통계 산출 |
| `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` | 학습 데이터 (토너먼트 라운드 정보 포함) |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 피처 추출 |
| `SampleSubmissionStage1.csv` | 제출 양식 (519,144행) |

> `DetailedResults`, `MasseyOrdinals`, `TeamCoaches` 등 **미사용** — Notebook 1 대비 매우 단순한 데이터셋

### 2.2 전처리 및 피처 엔지니어링 (Feature Engineering)

> **총 16개 피처** (팀별 raw 피처 × 차이/비율/교호작용)

#### A. Seed 피처 (`engineer_seed_features`) — 원본 5개 (T1, T2 각각)

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Seed_Num` | `Seed.str[1:3].astype(int)` | 시드 번호 (1~16) |
| `Seed_Strength` | `17 − Seed_Num` | 시드 강도 (높을수록 강함) |
| `Seed_Value` | `1 / Seed_Num` | 시드 역수 (비선형 변환) |
| `Seed_Squared` | `Seed_Num ** 2` | 시드 제곱 (비선형 변환) |
| `Seed_Percentile` | `(17 − Seed_Num) / 16` | 시드 백분위 (0~1) |
| `Seed_Region` | `Seed.str[0]` | 지역 문자 (W/X/Y/Z) — 피처로는 미사용 |

#### B. 팀 시즌 통계 (`compute_team_stats`) — 원본 5개 (T1, T2 각각)

`CompactResults`에서 승자(Winner)/패자(Loser) 관점의 집계를 각각 계산 후 `outer merge`:

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Games` | `Games_won + Games_lost` | 총 경기 수 |
| `Avg_pts_scored` | `(Pts_scored_w + Pts_scored_l) / Games` | 평균 득점 |
| `Avg_pts_allowed` | `(Pts_allowed_w + Pts_allowed_l) / Games` | 평균 실점 |
| `Avg_margin` | `(Margin_w + Margin_l) / Games` | 평균 마진 (Margin = WScore − LScore) |
| `Win_pct` | `Games_won / Games` | 승률 |

> 결측값 처리: 팀 스탯은 **전체 중앙값(median)**, 시드는 **최약 시드 기본값** (Seed_Num=16, Strength=1 등)

#### C. 매치업 피처 (16개 — `build_dataset`)

학습 데이터 구성 시 `build_dataset` 함수 내에서 생성:

**차이 피처 (Differentials, 5개):**
| 변수명 | 산출식 |
|--------|--------|
| `Seed_Num_Diff` | `Team1_Seed_Num − Team2_Seed_Num` |
| `Seed_Strength_Diff` | `Team1_Seed_Strength − Team2_Seed_Strength` |
| `Seed_Value_Diff` | `Team1_Seed_Value − Team2_Seed_Value` |
| `Avg_pts_scored_Diff` | `Team1_Avg_pts_scored − Team2_Avg_pts_scored` |
| `Avg_pts_allowed_Diff` | `Team1_Avg_pts_allowed − Team2_Avg_pts_allowed` |

**비율 피처 (2개):**
| 변수명 | 산출식 |
|--------|--------|
| `Seed_Num_Ratio` | `Team1_Seed_Num / (Team2_Seed_Num + 1)` |
| `Seed_Strength_Ratio` | `Team1_Seed_Strength / (Team2_Seed_Strength + 1)` |

**추가 차이/교호작용 (4개):**
| 변수명 | 산출식 |
|--------|--------|
| `Avg_margin_Diff` | `Team1_Avg_margin − Team2_Avg_margin` |
| `Win_pct_Diff` | `Team1_Win_pct − Team2_Win_pct` |
| `Seed_Num_Product` | `Team1_Seed_Num × Team2_Seed_Num` |
| `Seed_Sum` | `Team1_Seed_Num + Team2_Seed_Num` |

**시드 티어 지표 (3개):**
| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Same_Tier_Elite` | `(T1_Tier_Elite=1) & (T2_Tier_Elite=1)` | 양팀 모두 1~4시드 |
| `Same_Tier_Low` | `(T1_Tier_Low=1) & (T2_Tier_Low=1)` | 양팀 모두 13~16시드 |
| `Tier_Gap` | `abs(T1_Seed_Num//4 − T2_Seed_Num//4)` | 시드 단위 등급 차이 |

> 티어 구분: Elite(1~4), Contender(5~8), Mid(9~12), Low(13~16) — 각 팀별 4개 One-Hot 생성되나 최종 피처 목록에는 포함 안 됨

**기타 (2개):**
| 변수명 | 산출식 |
|--------|--------|
| `Is_Neutral` | 중립 코트 여부 (토너먼트=1) |
| `Round` | DayNum 기반 라운드 추정 (0=정규시즌, 1~6=토너먼트) |

> `get_round()`: DayNum < 136→0, <140→1(Round of 64), <145→2(Round of 32), <150→3(Sweet 16), <155→4(Elite 8), <160→5(Final Four), else 6(Championship)

### 2.3 학습 데이터 구성

#### Data Augmentation (승/패 대칭)
- 모든 정규시즌 + 토너먼트 경기에서 **원본 row (Win=1)** + **WTeamID↔LTeamID 교환 row (Win=0)** 생성
- 결과: Men 393,646행, Women 281,650행 (정규시즌+토너먼트 전체, 2배)
- **주의:** Notebook 1과 달리 정규시즌 경기도 학습 데이터에 포함됨

#### 특징
- 시간 기반 분할 **미사용** — StratifiedKFold로 무작위 분할
- Symmetric augmentation과는 다른 방식 (T1/T2 고정 순서 없이 승자=T1로 학습)

### 2.4 하이퍼파라미터 최적화 (Optuna)

**Speed Mode 설정:** `balanced` → 20 trials (fast=0, accurate=50)

각 모델별 성별별 개별 튜닝 (총 6회 최적화):

**LightGBM (20 trials, 3-fold StratifiedKFold):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `num_leaves` | 31 ~ 1,023 |
| `learning_rate` | 0.005 ~ 0.1 (log) |
| `feature_fraction` | 0.5 ~ 1.0 |
| `bagging_fraction` | 0.5 ~ 1.0 |
| `bagging_freq` | 1 ~ 10 |
| `min_data_in_leaf` | 10 ~ 300 |
| `lambda_l1` | 1e-8 ~ 10.0 (log) |
| `lambda_l2` | 1e-8 ~ 10.0 (log) |
| `min_gain_to_split` | 0.0 ~ 1.0 |

**XGBoost (20 trials, 3-fold StratifiedKFold):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `learning_rate` | 0.005 ~ 0.1 (log) |
| `max_depth` | 4 ~ 12 |
| `min_child_weight` | 1 ~ 10 |
| `subsample` | 0.5 ~ 1.0 |
| `colsample_bytree` | 0.5 ~ 1.0 |
| `reg_alpha` | 1e-8 ~ 10.0 (log) |
| `reg_lambda` | 1e-8 ~ 10.0 (log) |

**CatBoost (20 trials, 3-fold StratifiedKFold):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `learning_rate` | 0.005 ~ 0.1 (log) |
| `depth` | 4 ~ 10 |
| `l2_leaf_reg` | 1 ~ 10 |
| `border_count` | 32 ~ 255 |

- **CV 전략 (Optuna 내부):** `StratifiedKFold(n_splits=3)`, Early Stopping 50
- **목적 함수:** Log Loss 최소화
- **GPU 지원:** LightGBM → `device='gpu'`, XGBoost → `tree_method='hist', device='cuda'`, CatBoost → `task_type='GPU'`

**Pre-tuned 파라미터 (Optuna 스킵 시 사용, fast 모드):**
- 남성 LGB: `num_leaves=206, lr=0.0485, min_data_in_leaf=155`
- 남성 XGB: `lr=0.01576, max_depth=6, n_estimators=2000`
- 여성 LGB: `num_leaves=506, lr=0.0489, min_data_in_leaf=66`

### 2.5 모델 학습 (5-fold Stratified CV)

`train_model_cv` 함수로 각 모델 × 성별별 5-fold StratifiedKFold 학습:

- **LightGBM:** `lgb.train()` + Early Stopping 100, `num_boost_round=100` (Optuna에서 선택된 파라미터)
- **XGBoost:** `xgb.XGBClassifier(**params).fit()` + `eval_set` early stopping
- **CatBoost:** `CatBoostClassifier(**params).fit()` + `eval_set`

**실행 결과:**

| 모델 | Men CV LogLoss | Men CV Accuracy | Women CV LogLoss | Women CV Accuracy |
|------|---------------|-----------------|-----------------|-------------------|
| LightGBM | 0.51234 ± 0.00147 | 0.74425 ± 0.00076 | 0.46675 ± 0.00132 | 0.77264 ± 0.00110 |
| XGBoost | 0.57325 ± 0.00054 | 0.74400 ± 0.00094 | 0.51486 ± 0.00079 | 0.77205 ± 0.00111 |
| CatBoost | 0.51114 ± 0.00138 | 0.74493 ± 0.00077 | 0.46564 ± 0.00155 | 0.77296 ± 0.00100 |

### 2.6 Stacking Meta-Model (앙상블)

```python
# OOF 예측값 3개를 입력으로 사용
meta_X = np.column_stack([oof_lgb, oof_xgb, oof_cb])   # shape: (N, 3)
meta_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
meta_model.fit(meta_X, y)
```

- 3개 모델의 OOF(Out-of-Fold) 확률 예측값을 **새로운 3차원 피처**로 활용
- Logistic Regression이 각 모델에 대한 최적 가중치를 학습
- **Stacking LogLoss:** Men=0.51529, Women=0.47201

### 2.7 예측 및 제출

1. **성별 분리:** 시드가 양팀 모두 존재 → Men's (519,144행), 나머지 → Women's (0행)
   - **문제점:** 여성 구분 로직이 시드 존재 여부에만 의존 → 실제 결과에서 Women=0건
2. **Base 예측:** 5-fold 모델의 평균 예측값 활용
   ```python
   fold_pred += model.predict(X_subset) / len(model_list)   # 5-fold 평균
   ```
3. **Meta 예측:** `meta_model.predict_proba(base_predictions)[:, 1]`
4. **클리핑:** `np.clip(pred, 0.01, 0.99)` — Notebook 1(0.025~0.975)보다 넓은 범위

**제출 통계:**
- Min: 0.0733, Max: 0.9267
- Mean: 0.4980, Std: 0.2930

### 2.8 Notebook 1과의 주요 차이점

| 항목 | Notebook 1 | Notebook 2 |
|------|-----------|-----------|
| 피처 수 | 99개 | 16개 |
| 데이터 소스 | Detailed + Compact + Massey + Coach | Compact + Seeds만 |
| 학습 데이터 | 토너먼트 경기만 (~2,200행) | **정규시즌+토너먼트 전체** (~393K행) |
| 분할 전략 | 시간 기반 (2008~2022 Train / 2023~2025 Val) | StratifiedKFold (무작위) |
| 앙상블 방식 | Inverse-Brier 가중 평균 | Stacking (LogReg) |
| 보정 | Isotonic Calibration | 없음 |
| 피처 선택 | Shadow (Boruta) 방식 | 없음 |
| Augmentation | T1↔T2 대칭 (토너먼트만) | Winner/Loser 반전 (전체 경기) |

---

## 3. march-mania.ipynb
**주요 특징:** 188개 피처 + Optuna TPE 튜닝 → 3-Layer 다중 모델 앙상블 (FFM + GBDT + LR + NN → MoE 전문가 → 자동 최적 융합 선택) + Temporal OOF 검증

### 3.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` | Elo 산출, 팀 상세 통계 산출 |
| `MNCAATourneyDetailedResults.csv` / `WNCAATourneyDetailedResults.csv` | 학습 데이터 (토너먼트 매치업) |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 피처 |
| `MMasseyOrdinals.csv` | Massey 랭킹 시스템 |
| `MTeamCoaches.csv` | 코치 토너먼트 승률 |
| `MTeamConferences.csv` / `WTeamConferences.csv` | 컨퍼런스 정보 |
| `SampleSubmissionStage1.csv` | 제출 양식 |

> Notebook 1, 2와 달리 **DetailedResults + Coach + Conference** 등 가장 다양한 데이터 소스 활용

### 3.2 전처리 및 피처 엔지니어링 (Feature Engineering)

> **총 188개 피처** — 6개 그룹 × T1/T2 차이 + 교호작용

#### A. Elo Rating (`compute_elo`) — 3개 파생 피처

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `FinalElo` (→ `Elo1`, `Elo2`) | 아래 수식 기반 | 시즌 최종 Elo 점수 |
| `EloDiff` | `Elo1 − Elo2` | Elo 차이 |
| `EloWinProb` | `1 / (1 + 10^(−EloDiff/400))` | Elo 기반 승률 |

**Elo 업데이트 수식:**
```
ew = 1 / (1 + 10^((Elo_L − Elo_W) / 400))       # 승리 기대값
mov = ln(|WScore − LScore| + 1) × (2.2 / (ew × 0.001 + 1.0))   # 마진 승수
d = K × mov × (1 − ew)                             # 업데이트 크기
Elo_W += d,  Elo_L −= d
```
- **초기값:** 1500, **K:** 20
- **시즌 간 계승:** `Elo = 1500 + carry_pct × (Elo − 1500)`, `carry_pct = 0.75`
- **마진 승수(mov):** 단순 K가 아닌, 점수 차이의 log에 승리 기대값 역보정한 승수 적용

#### B. 고급 팀 통계 (`compute_team_stats`) — 21개 경기당 지표 × mean/std = 42개 + Recency 8개 + SOS 1개 = 51개 (T1, T2 각각)

**DetailedResults** 양측 관점(Winner/Loser → 각각 한 행)으로 분리 후 경기당 지표 산출:

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Poss` | `FGA − OR + TO + 0.44 × FTA` | 포제션 |
| `OPoss` | `OppFGA − OppOR + OppTO` | 상대 포제션 |
| `eFG` | `(FGM + 0.5 × FGM3) / FGA` | 유효 슈팅률 |
| `TS` | `Score / (2 × (FGA + 0.44 × FTA))` | True Shooting% |
| `3PAR` | `FGA3 / FGA` | 3점 시도 비율 |
| `FTR` | `FTA / FGA` | 프리스로 비율 |
| `AstR` | `Ast / Poss` | 어시스트율 |
| `TOVR` | `TO / Poss` | 턴오버율 |
| `ORR` | `OR / (OR + OppFGA − OppFGM)` | 공격 리바운드율 |
| `DRR` | `DR / (DR + OppOR)` | 수비 리바운드율 |
| `OffRtg` | `Score / Poss × 100` | 공격 레이팅 |
| `DefRtg` | `OppScore / Poss × 100` | 수비 레이팅 |
| `NetRtg` | `OffRtg − DefRtg` | 순 레이팅 |
| `Pyth` | `Score^11.5 / (Score^11.5 + OppScore^11.5)` | 피타고리안 승률 기댓값 |
| `Margin` | `Score − OppScore` | 점수 마진 |
| `FG2Pct` | `(FGM − FGM3) / (FGA − FGA3)` | 2점 슈팅률 |
| `FG3Pct` | `FGM3 / FGA3` | 3점 슈팅률 |
| `FTPct` | `FTM / FTA` | 자유투율 |
| `BlkR` | `Blk / (OppFGA − OppFGM3)` | 블록률 |
| `StlR` | `Stl / OPoss` | 스틸률 |
| `Win` | 1.0(승) / 0.0(패) | 승리 여부 |

**집계:** 각 21개 지표에 대해 시즌 전체 `mean`, `std` → **42개 컬럼** (예: `Win_mean`, `Win_std`, `Score_mean`, `eFG_std` 등)

**Recency 통계 (마지막 14경기):**
| 변수명 | 설명 |
|--------|------|
| `Rec_Win` | 최근 14경기 승률 |
| `Rec_NetRtg` | 최근 14경기 순 레이팅 |
| `Rec_Margin` | 최근 14경기 평균 마진 |
| `Rec_eFG` | 최근 14경기 유효 슈팅률 |
| `Rec_OffRtg` | 최근 14경기 공격 레이팅 |
| `Rec_DefRtg` | 최근 14경기 수비 레이팅 |
| `Rec_TOVR` | 최근 14경기 턴오버율 |
| `Rec_ORR` | 최근 14경기 공격 리바운드율 |

**SOS (Strength of Schedule):**
| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `SOS_OppDefRtg` | 상대 전체의 `DefRtg` 평균 | 상대 수비 강도 지표 |

#### C. Massey 랭킹 (`compute_massey`) — 4개 (T1, T2 각각)

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `MasseyMean` | RankingDayNum ≥ 128인 `OrdinalRank`의 평균 | 시즌 후반 평균 순위 |
| `MasseyMin` | 위의 최소값 | 최고 순위 |
| `MasseyMedian` | 위의 중앙값 | 중앙 순위 |
| `MasseyStd` | 위의 표준편차 | 순위 변동성 |

#### D. 코치 기록 (`compute_coach_records`) — 3개

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `CoachTWR1` | 코치의 역대 토너먼트 `TWins / (TWins + TLosses)` | T1 코치 토너먼트 승률 |
| `CoachTWR2` | 동일 | T2 코치 토너먼트 승률 |
| `D_CoachTWR` | `CoachTWR1 − CoachTWR2` | 코치 승률 차이 |

> 코치 미매핑 시 기본값 0.5 (중립)

#### E. 역사 대전 기록 (`compute_h2h`) — 2개

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `H2H_Games` | T1 vs T2 역대 토너먼트 교전 횟수 | 교전 경험 |
| `H2H_WinRate` | T1이 이긴 비율 (기본값 0.5) | 역대 승률 |

#### F. 컨퍼런스 + 시드 + 교호작용 피처 (`build_features`)

**시드 파생 (4개):**
| 변수명 | 산출식 |
|--------|--------|
| `SeedDiff` | `Seed1 − Seed2` |
| `SeedSum` | `Seed1 + Seed2` |
| `SeedWinProb` | `1 / (1 + exp(0.4 × SeedDiff))` |
| `UpsetPot` | `int(Seed1 > Seed2)` — 업셋 가능 지표 |

**컨퍼런스 (1개):**
| 변수명 | 설명 |
|--------|------|
| `SameConf` | 양팀 동일 컨퍼런스 여부 (0/1) |

**교호작용 피처 (6개):**
| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Elo_x_SeedDiff` | `EloDiff × SeedDiff` | Elo·시드 교차 |
| `NetRtg_x_SeedDiff` | `D_NetRtg_mean × SeedDiff` | NetRtg·시드 교차 |
| `Elo_x_NetRtg` | `EloDiff × D_NetRtg_mean` | Elo·NetRtg 교차 |
| `Massey_x_Elo` | `MD_MasseyMean × EloDiff` | Massey·Elo 교차 |
| `UpsetScore` | `|SeedDiff| × (1 − |EloDiff| / 200)` | 업셋 가능성 스코어 |
| `HotnessScore` | `|SeedDiff| × D_Win_mean` | 최근 폼 기반 업셋 위험 |

**차이 피처 (D_ prefix):** 모든 팀 통계(51개 + Massey 4개)에 대해 `T1_c − T2_c` 자동 생성

### 3.3 데이터 전처리

- **결측치:** `SimpleImputer(strategy='median')`
- **정규화:** `StandardScaler()` (평균 0, 분산 1)
- **학습 데이터:** T1 = min(WTeamID, LTeamID), T2 = max → `Label = int(WTeamID == T1)`
- **데이터 형상:** (2,410행 × 188 피처), 정샘플 비율: 0.502

### 3.4 하이퍼파라미터 최적화 (Optuna TPE)

`temporal_cv_score` 함수로 **시간 기반 순차 CV** (과거 시즌으로 학습 → 현재 시즌 예측):

**LightGBM (60 trials):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `num_leaves` | 8 ~ 128 |
| `max_depth` | 3 ~ 8 |
| `learning_rate` | 0.01 ~ 0.15 (log) |
| `min_child_samples` | 10 ~ 80 |
| `subsample` | 0.5 ~ 1.0 |
| `colsample_bytree` | 0.5 ~ 1.0 |
| `reg_alpha` | 1e-3 ~ 10.0 (log) |
| `reg_lambda` | 1e-3 ~ 20.0 (log) |
| `min_split_gain` | 0.0 ~ 1.0 |

**XGBoost (50 trials):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `max_depth` | 3 ~ 8 |
| `learning_rate` | 0.01 ~ 0.15 (log) |
| `subsample` | 0.5 ~ 1.0 |
| `colsample_bytree` | 0.5 ~ 1.0 |
| `gamma` | 0.0 ~ 5.0 |
| `min_child_weight` | 1 ~ 20 |
| `reg_alpha` | 1e-3 ~ 10.0 (log) |
| `reg_lambda` | 1e-3 ~ 20.0 (log) |

**CatBoost (40 trials):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `depth` | 3 ~ 8 |
| `learning_rate` | 0.01 ~ 0.15 (log) |
| `l2_leaf_reg` | 1.0 ~ 50.0 (log) |
| `bagging_temperature` | 0.0 ~ 2.0 |
| `random_strength` | 0.0 ~ 2.0 |
| `border_count` | 32 ~ 255 |

**LogisticRegression (30 trials):**
| 파라미터 | 탐색 범위 |
|----------|-----------|
| `C` | 1e-3 ~ 10.0 (log) |
| `penalty` | l1 또는 l2 |

**실행 결과:**
| 모델 | 최적 LogLoss |
|------|-------------|
| LightGBM | 0.07947 |
| XGBoost | 0.08436 |
| CatBoost | 0.08406 |
| LogReg | 0.08195 |

### 3.5 모델 아키텍처: 3-Layer 앙상블

#### Layer 1: Base Models (11개)

| 모델 | 개수 | 핵심 설정 | Data Augmentation |
|------|------|-----------|-------------------|
| **FastFFM** | 3개 | k=6, 15 epochs, batch=256 | 각각 다른 전략: ①80% 부트스트랩 ②70% 피처 서브셋 ③90% 비복원 추출 |
| **CatBoost** | 1개 | iterations=300, Optuna 파라미터 | 없음 |
| **LogisticRegression** | 3개 | C=Optuna, seed={42,123,456} | 없음 |
| **XGBoost** | 1개 | n_estimators=300, Optuna 파라미터 | 노이즈 증강 (σ=0.02) + 30% 반전(flip) |
| **LightGBM** | 1개 (또는 2개) | n_estimators=400, Optuna 파라미터 | 노이즈 증강 (동일) |
| **Neural Network** | 1개 | 3층 (128→64→32), dropout=0.3, Label Smoothing=0.05 | 없음 |

**FFM (Fast Factorization Machine) 상세:**
```python
# 예측 수식
score = w0 + Σ(w[i] × x[i])                    # 선형항
interactions = Σ Σ <v[i,f(j)], v[j,f(i)]> × x[i] × x[j]  # 교호작용항
pred = 1 / (1 + exp(-score - 0.1 × interactions))
```
- 피처당 최대 100개 비영 값만 사용 (차원축소)
- 예측 시 최대 50개 교호작용 항 계산
- 학습 시 랜덤 20개 교호작용 샘플링 (확률적 경사하강)
- 학습률 감쇠: `lr / √(epoch + 1)`

**Data Augmentation (XGBoost/LightGBM용):**
- 가우시안 노이즈: `X + N(0, 0.02)` (원본 크기 1배 추가)
- 부호 반전: 30% 샘플의 피처를 `-X`로, 라벨을 `1 - y`로 반전
- 총 2.3배 크기: `원본 → 원본 + 노이즈 + 반전` (예: 2410 → 5543행)

**NN Label Smoothing:** `y_smooth = y × 0.95 + 0.5 × 0.05` — 과적합 방지

#### Layer 2: MoE 스타일 이중 경로 융합

**Path 1 (은닉 요인 융합):**
- 입력: FFM 예측(3) + NN 은닉층 출력(32) + LR 예측(3) + XGB(1) + LGB(1) + NN(1) + CatBoost(1)
- 모델: NN (64→32, dropout=0.2, 20 epochs) + XGBoost (100 trees, depth=5)

**Path 2 (Pctr 점수 융합):**
- 입력: FFM 평균(1) + LR 평균(1) + XGB(1) + LGB(1) + NN(1) + CatBoost(1) = 6차원
- 모델: NN (32→16, dropout=0.2, 20 epochs) + XGBoost (80 trees, depth=4) + LogReg (C=0.5)

**3종 전문가 모델 (Expert Models):**
- Path1 입력 + SeedDiff + EloDiff 확장 피처 사용
- **General Expert:** XGBoost (60 trees, depth=4) — 전체 데이터
- **Upset Expert:** XGBoost (50 trees, depth=3) — `|SeedDiff| > 4 or |EloDiff| > 50` 가중
- **Favorite Expert:** XGBoost (50 trees, depth=3) — `|SeedDiff| > 6 or |EloDiff| > 80` 가중

#### Layer 3: MoE 게이팅 + 최적 융합 자동 선택

7가지 융합 방식을 동시 평가 → AUC 최고 방식 자동 선택:

| 방식 | 산출 방법 |
|------|-----------|
| **MoE 게이팅** | 조건별 전문가 가중: Upset시 (0.6/0.3/0.1), Favorite시 (0.1/0.3/0.6), 일반 (0.2/0.6/0.2) |
| **Rank 융합** | 5개 Path 예측의 rank → 평균 → 정규화 |
| **선형 융합** | `SLSQP` 최적화 (엔트로피 정규화 + 5개 random seed 재시작) → 가중합 |
| **조합 융합** | 0.5 × 선형 + 0.5 × MoE |
| **Stacking-LogReg** | 8개 예측(Path1/2 × NN/XGB + Path2-LR + 3 Expert) → LogReg C=1.0 |
| **Stacking-Ridge** | 동일 입력 → RidgeClassifier α=1.0 → sigmoid |
| **Stacking 평균** | (LogReg + Ridge + MoE) / 3 |

**Isotonic Calibration:** 최종 선택된 예측에 `IsotonicRegression(out_of_bounds='clip')` 적용 → `clip(0.025, 0.975)`
- 보정 AUC ≥ 원본 AUC인 경우에만 사용 (자동 판정)

### 3.6 검증: Temporal OOF (시간 순차 검증)

2003~2025 시즌 중, 각 시즌을 순차적으로 test set으로 사용 (이전 시즌 모두 train):

| 시즌 | Train/Test | AUC | LogLoss |
|------|-----------|------|---------|
| 2005 | 128/64 | 0.9589 | 0.2144 |
| 2006 | 192/64 | 0.9268 | 0.3171 |
| 2007 | 256/64 | 0.9853 | 0.0859 |
| 2008 | 320/64 | 0.9990 | 0.0733 |
| 2009 | 384/64 | 0.9833 | 0.1335 |
| 2010 | 448/127 | 0.9764 | 0.1182 |
| 2011 | 575/130 | 0.9537 | 0.2015 |
| 2012 | 705/130 | 0.9776 | 0.1099 |
| 2013 | 835/130 | 0.9494 | 0.2367 |
| 2014 | 965/130 | 0.9469 | 0.2525 |
| 2015 | 1095/130 | 0.9839 | 0.1215 |
| 2016 | 1225/130 | 0.9585 | 0.1916 |
| 2017 | 1355/130 | 0.9614 | 0.1870 |
| 2018 | 1485/130 | 0.9692 | 0.1380 |
| 2019 | 1615/130 | 0.9836 | 0.1417 |
| 2021 | 1745/134 | 0.9761 | 0.1388 |
| 2022 | 1879/134 | 0.9840 | 0.1090 |
| 2023 | 2013/134 | 0.9488 | 0.2167 |
| 2024 | 2147/134 | 0.9729 | 0.1351 |
| 2025 | 2276/134 | 0.9923 | 0.0546 |
| **종합** | **2282/2410** | **0.9686** | **0.1578** |

> 2020 시즌 = COVID 취소로 데이터 없음

### 3.7 예측 및 제출

- **전량 학습:** 전체 2,410행으로 최종 모델 학습
- **전량 AUC:** 1.000000 (완전 과적합 — 학습 데이터 기준)
- **클리핑:** `clip(0.025, 0.975)`

**제출 통계:**
- Mean: 0.378, Std: 0.451
- Min: 0.025, Max: 0.975

### 3.8 주요 설계 특징 및 한계점

**장점:**
- 가장 풍부한 피처 엔지니어링 (188개, 6개 데이터 소스)
- Temporal CV로 미래 데이터 누수 방지
- MoE 전문가 시스템 (Upset/Favorite 상황별 최적화)
- 7가지 융합 방식 자동 선택

**한계점:**
- 학습 시 내부 Train AUC가 항상 1.0 → **Layer 2/3에서 과적합 위험** (학습 데이터로 자기 자신 예측)
- OOF에서 `is_upset`, `is_favorite` 마스크가 항상 `[0, 0, 0]` → **Expert 모델 미분화** (SeedDiff/EloDiff가 입력 피처에서 정규화되어 원본 스케일 소실)
- FFM 구현이 비효율적 (Python 루프 기반, 대규모 데이터에 부적합)
- `MTeamCoaches`만 사용하여 **여성 부문 코치 데이터 미적용**

---

## 4. march-ml-mania-2026-lgbm-xgb-catboost.ipynb
**주요 특징:** 최소한의 피처(13개) + 고정 하이퍼파라미터 + LGB/XGB/CatBoost 단순 평균 앙상블 — 가장 간결한 베이스라인

### 4.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | 팀 시즌 통계 산출 |
| `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` | 학습 데이터 (토너먼트 매치업) |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 번호 |
| `MMasseyOrdinals.csv` | MOR 랭킹 **(남성만)** |
| `MTeams.csv` / `WTeams.csv` | 남녀 팀 ID 구분용 |
| `SampleSubmissionStage1.csv` | 제출 양식 |

> `DetailedResults`, `TeamCoaches`, `TeamConferences` 등 **미사용** — 6개 노트북 중 가장 단순한 데이터셋

### 4.2 전처리 및 피처 엔지니어링 (Feature Engineering)

> **총 13개 피처** (수치 11 + 카테고리 2)

#### A. 정규시즌 팀 통계 (`build_season_stats`) — 4개 (T1, T2 각각)

승자/패자 행을 각각 분리 후, 팀별·시즌별 집계:

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Games` | `count(Win)` | 총 경기 수 (피처 미사용, 내부 계산용) |
| `Wins` | `sum(Win)` | 승리 수 (피처 미사용) |
| `AvgScore` | `mean(ScoreFor)` | 평균 득점 |
| `AvgMargin` | `mean(Margin)`, `Margin = ScoreFor − ScoreAgainst` | 평균 점수 마진 |
| `WinRate` | `Wins / Games` | 승률 |

> `ScoreFor`: 승자일 때 `WScore`, 패자일 때 `LScore`. `ScoreAgainst`: 반대.

#### B. 시드 번호 (`parse_seed`) — 카테고리 2개

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `T1_Seed` | `int(''.join(filter(str.isdigit, Seed)))` | 시드 번호 (**카테고리형**) |
| `T2_Seed` | 동일 | 시드 번호 (**카테고리형**) |

> 결측시 `-1`로 대체 후 `int → category` dtype. 단순 문자열에서 숫자만 추출 (예: `W01` → `1`, `X16a` → `16`)

#### C. Massey Ordinals (MOR) — 1개 (T1, T2 각각)

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `T1_MOR` / `T2_MOR` | `SystemName == 'MOR'`인 행만 필터 → 시즌/팀 별 마지막 DayNum의 `OrdinalRank` | MOR 랭킹 순위 |

> **남성 부문에만 적용** — 여성 부문은 `mor_df=None`으로 호출하여 `MOR = NaN`

#### D. 매치업 차이 피처 (`build_train_df`) — 수치 5개

T1 = min(WTeamID, LTeamID), T2 = max로 정렬 후 차이 계산:

| 변수명 | 산출식 |
|--------|--------|
| `SeedDiff` | `T1_Seed − T2_Seed` |
| `WinRateDiff` | `T1_WinRate − T2_WinRate` |
| `MarginDiff` | `T1_Margin − T2_Margin` |
| `ScoreDiff` | `T1_Score − T2_Score` |
| `MORDiff` | `T1_MOR − T2_MOR` |

#### E. 최종 피처 목록

```python
NUM_COLS = ['SeedDiff', 'WinRateDiff', 'MarginDiff', 'ScoreDiff', 'MORDiff',
            'T1_WinRate', 'T2_WinRate', 'T1_Margin', 'T2_Margin', 'T1_MOR', 'T2_MOR']
CAT_COLS = ['T1_Seed', 'T2_Seed']
FEAT_COLS = NUM_COLS + CAT_COLS   # 총 13개
```

### 4.3 학습 데이터 구성

- **라벨:** `Label = int(WTeamID == T1)` — T1이 이기면 1, T2가 이기면 0
- **결측치:** 수치 피처는 global median으로 대체, `SeedDiff`나 `Label`이 NaN인 행은 drop
- **남녀 분리:** 독립적으로 학습/예측 (`m_train`, `w_train` 별도 구성)
- **Augmentation 없음** — 원본 토너먼트 매치업만 사용

### 4.4 모델 학습 (고정 하이퍼파라미터)

> **하이퍼파라미터 튜닝 없음** — 모든 파라미터 하드코딩

**LightGBM:**
| 파라미터 | 값 |
|----------|-----|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `min_child_samples` | 20 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

**XGBoost:**
| 파라미터 | 값 |
|----------|-----|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 5 |
| `min_child_weight` | 20 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `tree_method` | 'hist' |
| `enable_categorical` | True |

**CatBoost:**
| 파라미터 | 값 |
|----------|-----|
| `iterations` | 300 |
| `learning_rate` | 0.05 |
| `depth` | 5 |
| `task_type` | 'CPU' |

> 세 모델 모두 `random_state=42`, `n_jobs=-1`(또는 `thread_count=-1`)

### 4.5 교차 검증 및 앙상블

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 각 fold에서:
prob = (prob_lgb + prob_xgb + prob_cat) / 3    # 단순 1/3 평균
ll = log_loss(y_va, prob)
```

- **CV 전략:** 5-fold Stratified KFold (무작위 분할, 시간 기반 아님)
- **앙상블 방식:** 3개 모델의 예측 확률 **단순 산술 평균** (1/3 균등 가중)
- **평가 지표:** Log Loss
- **실행 결과:** 출력 셀 없음 (노트북 미실행 상태)

### 4.6 예측 및 제출

1. **남녀 분리:** `T1`이 `MTeams.TeamID`에 존재하면 Men's, `WTeams.TeamID`에 존재하면 Women's
2. **제출 피처 추출:** `Parallel(n_jobs=-1, backend='threading')` + `tqdm`으로 병렬 처리
3. **전체 학습 데이터 사용:** CV가 아닌 train 전체로 3개 모델 학습 → 평균 예측
4. **클리핑:** `clip(0.025, 0.975)`

### 4.7 특이사항 및 비교

| 항목 | Notebook 4 (본 노트북) | 다른 노트북 |
|------|----------------------|-------------|
| 피처 수 | **13개** (최소) | 16~188개 |
| 데이터 소스 | Compact + Seeds + MOR | Detailed, Coach, Conference 등 |
| 하이퍼파라미터 | **고정 (튜닝 없음)** | Optuna/수동 튜닝 |
| 앙상블 방식 | **단순 1/3 평균** | Stacking, MoE, SLSQP 가중 등 |
| CV 전략 | StratifiedKFold (무작위) | Temporal, Time-series |
| 보정 | **없음** | Isotonic, Logit Calibration |
| MOR 여성 | **NaN** (남성만 사용) | - |
| 시드 처리 | **카테고리형** (unique) | 수치 + 파생 변수 |

> **평가:** 6개 노트북 중 가장 단순하여 베이스라인으로 적합. 하지만 ① MOR 여성 미적용, ② 시간 기반 검증 미사용, ③ Data Augmentation 없음, ④ 파라미터 최적화 없음의 한계가 있음.

---

## 5. [ncaa-2026-eda-elo-ratings-and-gradient-esemble.ipynb](file:///Users/swoo64/Desktop/March-Machine-Learning-Mania-2026/Example/ncaa-2026-eda-elo-ratings-and-gradient-esemble.ipynb)
**주요 특징:** 방대한 양의 EDA(시드, 홈어드밴티지, 점수 트렌드, 업셋 등)를 통해 도출된 통계적 구조를 그대로 피처 33개로 맵핑한 실용적 Baseline 접근. Brier Score에 맞춰 고정 하이퍼파라미터 모델 3개를 앙상블함.

### 사용 데이터 (Data Sources)
- **`*CompactResults.csv`**: 정규시즌(`RegularSeason`), NCAA 토너먼트(`NCAATourney`) 및 **`SecondaryTourney` (NIT, CBI, CIT, 등)**의 결과까지 병합하여 최초로 더 넓은 범위의 Elo Rating을 구축.
- **`*DetailedResults.csv`**: Four Factors 등 세부 오펜시브/디펜시브 스탯 추출용.
- **`*Seeds.csv`**: 시드 정보 매핑 기능. 결측치는 8.5로 보간 (`SEED_DEFAULT = 8.5`).
- **`*TeamConferences.csv`**: 컨퍼런스별 약강도 계산 용.
- **`MTeamCoaches.csv`**: 성인 남성팀 토너먼트 전용의 감독 출전 경험치/재직 기간 계산.
- *Massey Ordinals (MOR) 및 타 외부 레이팅 지표는 일절 사용하지 않음.*

### 데이터 전처리 및 피처 엔지니어링 (총 33개 파생변수)
각 피처들은 특정 연도의 주어진 팀(T1, T2)에 대하여, 모델이 각 팀의 값과 두 팀 간의 차이(`_d`)를 함께 입력받도록 설계되었습니다.

#### 5-1. Elo Rating System (`build_elo`)
1.  **초기값 및 상수:** 초기 Elo는 `1500`, K-factor는 기본 상수 파라미터 `K=20`을 이용.
2.  **Margin of Victory 곱셈기 (Multiplier):** 단순 승패를 넘어 점수차(마진)의 로그값을 K값에 곱하여 승리 규모를 반영.
    - `k_adj = 20 * np.log(max(abs(WinScore - LossScore), 1) + 1)`
3.  **홈 코트 어드밴티지 (Home Advantage):** 홈/원정 지표에 따라 상수 `HOME=100`을 부여하여 원정 승리에 상대적으로 더 높은 리워드를 줌.
    - `ha = 100` (Home 승리), `-100` (Away 승리), `0` (Neutral 승리)
    - 승리 기대확률 (Win Expectancy): `we = 1.0 / (1.0 + 10.0 ** ((elo_loser - elo_winner - ha) / 400.0))`
4.  **시즌 간 회귀 (Mean Reversion):** 새 시즌 돌입 시 전년도 종가를 75%만 이어가고, 25%는 평균치(1500)로 회귀 (`REV = 0.75`).
    - `elo[t] = 1500 * (1 - 0.75) + elo[t] * 0.75`
5.  **산출 피처:** `elo_d` (T1 Elo - T2 Elo), `elo1`, `elo2`.

#### 5-2. Four Factors 및 심화 지표 (20개 이상 피처)
`DetailedResults`를 입력받아 공격/수비 관련 박스스코어 수치들을 전부 누적(sum)한 후 경기 수(`n`) 등으로 나눠 생성. Dean Oliver의 4요소를 핵심 뼈대로 함.

1.  **포제션 계산 (Possessions):** `0.475` 계수를 곱한 FTA 활용.
    - 공격: `poss = fga - orb + to + 0.475 * fta`
    - 방어: `oposs = ofga - oorb + oto + 0.475 * ofta` (상대 스탯 기준)
2.  **효율성 스탯 (Efficiency):** 100 포제션 당 스코어로 환산.
    - `oeff = (pts / poss) * 100`
    - `deff = (ops / oposs) * 100` (수비 효율. 낮을수록 좋음)
    - `neff = oeff - deff` (순 효율) -> 이 값은 개별 피처 `neff1`, `neff2` 로도 활용됨.
3.  **Four Factors / 디테일 스탯 (Total 17개):**
    - `wpct`: 정규시즌 승률 (`wins / n`)
    - `margin`: 평균 득실 마진 (`(pts - ops) / n`)
    - `efg`: 유효슈팅률 (`(fgm + 0.5 * f3m) / fga`)  /  `oefg`: 디펜시브 유효슈팅률 억제
    - `tor`: 턴오버 비율 (`to / poss`)  /  `otor`: 수비 턴오버 창출률
    - `orpct`: 공격 리바 확률 (`orb / (orb + odrb)`)  /  `oorpct`: 수비 리바운드 허용률
    - `ftr`: 자유투 빈도 (`ftm / fga`)  /  `oftr`: 자유투 허용 빈도
    - `f3pct`, `of3pct`: 팀 및 상대팀 3점 슛 성공률
    - `astr`: 팀 어시스트 기반 (`ast / fgm`)
    - `stlpg`, `blkpg`, `drbpg`: 경기(n)당 스틸, 블록, 디비전리바운드 단순 스탯
    - `pace`: 경기 속도 (`(poss + oposs) / (2 * n)`)
    - 위 17개 지표에 대해 팀간 차이값인 `_d` 17개가 생성됨.

#### 5-3. 컨텍스트 피처 (Contextual Indicators)
1.  **Strength of Schedule (`sos_d`):** 상대했던 팀(정규시즌)들의 앞서 구한 **평균 Elo**를 계산.
    - 즉, 얼마나 강한 리그와 스케줄을 거쳤는지 난이도를 측정.
2.  **Late-Season Momentum (`mom_d`):** 정규시즌 맨 끝(DayNum 기준 오름차순) **최근 10경기** 승률.
3.  **Conference Strength (`conf_d`):** 해당 시즌 컨퍼런스 소속 전체 팀들의 **평균 Elo**.
4.  **Coach Experience (`cexp_d`, `cten_d`):** 남성 대회 한정 (여성은 데이터 없음 처리).
    - `cexp`: 해당 감독의 통산 NCAA 토너먼트 참가 횟수. (`cumcount`)
    - `cten`: 현재 팀 내 연속 재임 년수. (`cumcount`)
5.  **Seed (`seed_d`, `s1`, `s2`):** 두 팀의 순수 시드 번호 배정 및 시드 차이값. 결측치는 `8.5`.

*최종 모델 투입 변수는 `elo_d, elo1, elo2, seed_d, s1, s2, wpct_d, margin_d, oeff_d, deff_d...` 등 약 33개.*

### 학습 파이프라인 및 모델링 구조
*   **Temporal CV (엄격한 시간순 홀드아웃 격리):** 
    - 랜덤 K-Fold 방식을 폐기하고, **Train:** 남성(2003~2021) / 여성(2010~2021), **Validation:** 가장 최신의 과거 구간인 **2022년~2025년**을 사용하여 실제와 동일하게 검증.
*   **하이퍼파라미터 튜닝 생략 (Fixed Tuning):** Optuna 과정 없이 아래의 3개 그래디언트 부스팅 모델을 조기종료(Early Stopping)와 하드코딩된 파라미터로 학습.
    1.  **LightGBM** (Best iteration: 201)
        - `learning_rate=0.02, num_leaves=31, n_estimators=2000`
        - `min_child_samples=15, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0`
    2.  **XGBoost** (Best iteration: 404)
        - `learning_rate=0.02, max_depth=5, n_estimators=2000`
        - `min_child_weight=15, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, tree_method="hist"`
    3.  **CatBoost** (Best iteration: 694)
        - `learning_rate=0.02, depth=5, iterations=2000`
        - `min_data_in_leaf=15, l2_leaf_reg=3.0, subsample=0.8, eval_metric="Logloss"`

### 앙상블 및 Calibration 적용
1.  **단순 평균 로직:** 세 모델의 확률 아웃풋을 정확히 `1/3` 가중치로 합침. (`ens_pred = (p1 + p2 + p3) / 3.0`)
2.  **Probability Clipping:** 불확실성 패널티(Brier Score 특성)를 방어하기 위해 극한 스코어 생성 확률을 제한함. `np.clip(ens_pred, 0.02, 0.98)`
3.  **Retraining:** 추론용 Stage 2 Submission 생성시, 앞서 확보한 각각 모델들의 `Best Iteration` 값을 기준으로 **Train 셋(과거) + Validation 셋(22~25시즌)을 모두 병합하여 재학습(Full Fit)** 함. 

### 퍼포먼스 및 실제 통계
- 토너먼트 2022~2025 OOF(Validation set) 평가 (2,410건의 정규 경기).
- **LogLoss:** 0.34928   |   **Brier Score (SSE):** 0.10995   |   **ROC-AUC:** 0.9259
- **Expected Calibration Error (ECE):** 0.0256 (매우 잘 교정됨)
- **Top Feature Importance:** 
  1. `elo_d` (압도적 1위, 값: 847) 
  2. `wpct_d` (승률 갭 508)
  3. `elo1` & `elo2` (각 ~300)
  4. `seed_d` (시드 격차 271)
  5. `conf_d` (컨퍼런스 힘차이 225) 
  6. `ftr_d` (자유투빈도 격차 223)

### 종합 코멘트 (Review)
EDA에 엄청난 공수(코드 2400줄 중 1000줄 이상)를 들인 후, 이 과정에서 발견된 인사이트(업셋은 시드갭 클때 드묾, 스코어 인플레, 홈어드밴티지 명확함 유무 등)만을 직접적인 통계 수학 공식(피처)으로 구현해 트리 모델에 넘겼습니다.  
특히 1) **Secondary Tourney 정보**까지 편입해 확장시킨 Elo 시스템, 2) Data Leakage를 차단하는 **엄격한 시간순 Holdout 검증**, 이 두 가지 특성 때문에 Baseline으로 쓰기 아주 훌륭한 "실용주의적" 디자인 패턴이며, 복잡한 Optuna 앙상블 없이도 AUC 0.92 이상의 탄탄한 신뢰성을 지닙니다.

---

## 6. [strong-mania-2026-forcasting-4-model-ensemble.ipynb](file:///Users/swoo64/Desktop/March-Machine-Learning-Mania-2026/Example/strong-mania-2026-forcasting-4-model-ensemble.ipynb)
**주요 특징:** 퍼블릭 리더보드 오버피팅(과대적합)을 최대한 방어하고 "일반화된 실제 예측성능" 확보에 집중한 스위스칼 메스 같은 파이프라인. 남/녀 대회 데이터를 하나의 모델로 완전히 묶어서(`GenderFlag` 활용) 학습하며, 수백 개의 파생 변수를 Brier Score에 맞춰 최적 혼합(SLSQP & Logit Calibration)하는 것이 특징입니다.

### 사용 데이터 (Data Sources)
- **`*DetailedResults.csv`**: 정규시즌 상세 박스스코어를 가공하기 위한 핵심 재료. 
- **`*CompactResults.csv`**: 정규시즌, 토너먼트. (여타 모델과 달리 Secondary Tourney 기록은 쓰지 않음)
- **`*Seeds.csv`**: 매 시즌 시드 배정.
- **`*TeamConferences.csv`**: 팀별 소속 컨퍼런스를 바탕으로 해당 컨퍼런스의 전력을 구함.
- **`MMasseyOrdinals.csv`**: 남성 전용 데이터. 커버리지가 높은(8시즌 이상 데이터를 가진) 시스템만 선별한 뒤 통계 피처를 만듦.

### 데이터 전처리 및 피처 엔지니어링 (입력 변수는 총 344개)
`GenderFlag`를 이용해 남성을 `0`, 여성을 `1`로 처리하고 단일 훈련셋에서 복합 레이팅을 도출해냅니다. 피처 엔지니어링의 양이 방대하며 크게 4가지 단계로 요약됩니다.

#### 6-1. 기본 경기 스탯 추출 및 기초 스탯 (포제션 및 4요소)
남/여 시즌 `DetailedResults` 데이터를 승/패 Row가 아닌 팀별(Teams1~2 병합) Row 단위 게임으로 바꾸고 방대한 스탯을 산출합니다.
1.  **포제션(Possession) 계산 공식:** 0.475 상수 사용 (`Poss = FGA - OR + TO + 0.475 * FTA`)
2.  **Pace (경기 속도):** `Pace = 0.5 * (Poss + OppPoss)`
3.  **효율성(O/D/Net):**
    - `OffRtg = 100 * (PF / Poss)`
    - `DefRtg = 100 * (PA / OppPoss)`
    - `NetRtg = OffRtg - DefRtg`
4.  **4요소 비율 추출:** 팀 및 상태(Opp) 각각 생성 (e.g. `eFG`, `Opp_eFG` 등).
    - `eFG = (FGM + 0.5 * FGM3) / FGA`
    - `TOVPct = TO / Poss`
    - `ORBPct = OR / (OR + ODR)`
    - `FTR = FTM / FGA`

#### 6-2. 팀-시즌 별 스탯 누적 (단순, 변동, 후반 폼)
각 팀-시즌 별로 경기들을 누적합하여 단일 스탯 피처(`aggregate_team_features`)를 제작.
1.  **기초 평균 (`_mean` 16개):** `PF_mean`, `PA_mean`, `OffRtg_mean`, `DefRtg_mean`, `eFG_mean`, `NumOT_mean` 등.
2.  **안정성(변동폭) (`_std` 3개):** 경기마다 기복 측정용 `Margin_std`, `NetRtg_std`, `Pace_std`.
3.  **최근 폼 가중 평균 (`Recency _wmean` 13개):** 시즌 마지막에 가까울수록 기하급수적으로 큰 가중치를 주는 변수 개발.
    - 가중치 계수 함수: `RecencyW = np.exp((DayNum - 132.0) / 30.0)`
    - 위 `RecencyW`를 각각의 값에 곱하고 총합으로 나눈 가중 평균 (`예: NetRtg_wmean, Margin_wmean, eFG_wmean`)
4.  **마지막 스퍼트 (`_recent` 6개):** 시즌 극후반 (DayNum >= 118) 경기만 별도 필터링하여 평균 값 구함. (`Win_recent`, `Margin_recent` 등)
5.  **위치(Location) 승률:** 홈/어웨이/중립 각 승률 산출. (결측일 시 전체 승률(`WinPct`)로 대체).

#### 6-3. 복합 레이팅 및 스케줄 강도 보정 시스템
상대적인 강력함을 계산하기 위해 수학 모델(Elo, SRS, 상대전적 전이치)들을 활용.
1.  **SOS (Strength of Schedule):**
    - 상대했던 각 팀들에 대해 (해당 팀의 `최근 폼 기반 WinPct` × 당시 대결 `RecencyW` 가중치) 값들을 구해서 다시 누적하여 평균 구함 (`SOS_WinPct`).
    - 동일하게 상대의 NetRtg에 대응되는 값도 구함 (`SOS_NetRtg`).
    - **조정 효율 (AdjNetRtg):** 기본 넷 효율에서 상대했던 난이도를 차감 (`AdjNetRtg = NetRtg_wmean - SOS_NetRtg`)
2.  **Elo Rating System:**
    - 홈 어드밴티지 상수 가중치 `HOME = 80.0`. (K-Factor = 22.0)
    - 득실마진 승수(MOV Multiplier) 변환 공식: `((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * |rw - rl|)`
    - 시즌 리셋 회귀: 이전해 수치를 75%만 승계함 (`season_start = 0.75 * prior + 0.25 * 1500.0`).
    - Z-Score 적용 (`EloZ`).
3.  **SRS (Simple Rating System) 행렬 방정식:**
    - 팀간 득실 마진 차이를 `rating` 미지수로 두고 상대전적 엣지(Edge) 비율을 구하여 연립방정식으로 도출.
    - `M (전적비율 매트릭스)`을 구한 후, `A = I - M + 0.08*I (릿지정규화 0.08)` 하에서 파이썬 `np.linalg.solve(A, mov)` 를 풀어 고유 SRS를 획득. Day 110 이후 따로 떼어된 `SRS_recent` (릿지 0.12) 를 1개 더 산출.
4.  **Massy Ordinals (남성 전용):**
    - 8번 이상 랭킹 측정된 시스템만 분류. 평균(`MasseyRankMean`), 편차, 최고, 빈도 수 집계 후 Z스코어화(`MasseyStrengthZ` = `-MasseyRankMean` 후 분포정규화).
5.  **Power Composite (파워 지수합산) & 가상 시드(`ExpectedSeed`):**
    - `PowerComposite`: `AdjNetRtg_z`(0.30), `EloZ_z`(0.23), `SRS_z`(0.20), `WinPct_z`(0.12), `NetRtg_recent_z`(0.10), `Margin_recent_z`(0.05) 총 6개 파생 변수들의 고정 가중치 합.
    - `ExpectedSeed`: 이것들의 순위 위계를 정해, `1~68위` 안이면 4로 나눈 값(순수 예상시드치), 68위 밖이면 페널티 보간. 진짜 시드 번호(`SeedNum`)가 없으면 이 `ExpectedSeed`가 대체(`ModelSeed`). (시드 격차 계산시 유리).

#### 6-4. 매치업 블렌딩 (Matchup Construction) 총합 344개 피처.
토너먼트 경기를 앞둔 두 팀(T1, T2) 간에 각 `TeamID`에 속한 개별값 ~70개 피처 씩(`T1_`, `T2_`)이 병합됩니다(Total 140개). (※과거 히스토리가 없는 신입생급 팀은 `build_team_history_fallback` 으로 과거의 역대 성적 비례값으로 결측 대체함)
1. **모든 수치 피처간의 계산:** `_diff` (T1 - T2), `_absdiff` (절댓값) 총 140개 피처 추가 생성.
2. **합산(`_sum`):** 페이스나 넷레이팅 같은 볼륨 지표인 `Pace_wmean`, `WinPct`, `AdjNetRtg`, `ModelSeed`, `ConfPowerMean` 5개 대상 합산 피처 추가.
3. **직접 확률 모델(`EloWinProbNeutral`):** 위에서 본 T1 Elo, T2 Elo의 격차값을 확률공식 1개 그 자체 피처로 치환: `1.0 / (1.0 + np.power(10.0, -elo_diff / 400.0))` 
4. **`SameConference` (0 또는 1), T1 및 T2_ConfAbbrev 카테고리값 매핑** 등.

### 학습 및 검증 전략 (Rolling Seasons Validation)
- K-Fold 혹은 샘플 랜덤 추출 분가를 완전 배제하고 미래 누수 제로(0% Data Leakage) 전략 취함.
- `Season < S`로 훈련하고, 정확히 `S` 연도로 검증하는 **시계열 롤링 스플릿(Time-series Rolling OOF)** 전개. 최신 10회 검증: `[2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]` (2020년 미개최 취소).
- 평가 측정 항목: `brier_score_loss` (MSE 확률 버전). 

### 모델 블렌딩(앙상블) 및 확률 보정 (Calibration) 기술
1. **Base Models 파라미터 튜닝 생략(Holdout):**
   - Optuna 탐색을 포기하고 단순하지만 묵직한 지정 파라미터로 구성된 4개 모델 활용.
   - `LogisticRegression`: (elasticnet, L1_ratio=0.08, C=0.6, max_iter=6000, 딥러닝 보단 스케일링 회귀에 의존)
   - `LightGBM`: (lr=0.02, 1300 estimators, max_depth=-1, num_leaves=31) - Early Stopping 100 적용.
   - `XGBoost`: (lr=0.02, max_depth=4, 1400 estimators, hist 알고리즘) 
   - `CatBoost`: (lr=0.025, depth=6, 1000 iter, l2_leaf_reg=6.0)
   - `elo`: Base Model의 예측치인 양, 위 6-4에 만들어진 `EloWinProbNeutral` 피처 열을 고스란히 1개의 확률값 결과로 투입. (총 5개 아웃풋)
2. **SLSQP-optimized Weightings:** 앞서 도출한 OOF Validation 셋의 타겟 확률을 대상으로, 위 5개 요소가 어느 비율(가중치 합 `1.0`)로 합쳐져야 **Brier Score를 가장 낮출 수 있는지** SciPy 비선형 최소치 검색(`minimize(method="SLSQP")`)을 진행.
   - *실행 결과 가중치 (OOF):* CatBoost(91.3%), LightGBM(3.7%), Elo 확률점유율(3.2%), LR(0.9%), XGB(0.6%) - CatBoost가 Brier스코어 안정성이 압도적이라 가중치 쏠림이 있음.
   - 해당 점수들을 합산한 값을 예측 밴드 극한치 피하기를 위한 1차 클리핑(`0.001 ~ 0.999`).
3. **Logit Calibration:** 1차 클리핑 된 앙상블 블렌드 확률 `p`를 Logit 함수(승산 로그 `log(p / (1-p))`)로 1차원 선형 공간치 값으로 던져버린 뒤, 이 값을 피처로 한 번 더 `LogisticRegression(C=50.0)`에 넣고 타겟 변수(실제 승패 타겟)와 마지막 핏(Train)을 시켜 결과를 곡선화 보정(Extremes Smoothing)시킴.
4. **결과 Retraining & 최종 Clipping:** 전체 데이터 셋(Train+Valid)로 트리 사이즈를 다소 낮추어(LGB 900, XGB 1200, CAT 800) 재학습 후 `np.clip(final_pred, 0.02, 0.98)` 적용(과대신뢰 극단적 오류 제어). 최종 통산 OOF 캘리브레이션 Brier Score: `0.1642`.

### 종합 코멘트 (Review)
머신러닝 알고리즘 자체(하이퍼파라미터나 구조)를 건드리기보다, 본 대회에 맞춰 도메인 지식을 집어넣은 **수학적 레이팅 설계 아키텍처 (Recency 지수 디케이 역산, SRS 연립방정식 행렬 도출, Power Composite 가중 종합 시드 개발) 에 뼈를 갈아 넣은 피처 엔지니어링 끝판왕** 노트북 파이프라인입니다.

예측 성능 최적화 모델에 그치지 않고 목적함수(Brier Score Loss) 최소화 방정식(SLSQP)으로 가중치를 배분한 뒤 한 단계 더 Logit(로짓) 형태로 확률 기형(왜곡)까지 교정해 냅니다. 이는 리더보드에서 0.00 만점(퍼펙트 예측) 트릭을 쓰는 유저들의 평가를 의도적으로 무시하며, 실전 베팅을 위한 가장 논문 친화적이고 견고한 프레임워크 베이스 파이프라인입니다.

---

## 7. march-machine-learning-mania-2026-8.ipynb
**주요 특징:** Stage 1 평가 구조를 역이용한 **하이브리드 전략** — 과거 토너먼트에서 실제로 일어난 경기는 결과를 직접 조회(Lookup)하여 거의 완벽한 확률(0.999/0.001)을 부여하고, 실제로 일어나지 않은 매치업에만 ML 앙상블(XGB+LGB+CatBoost)로 예측. Elo + 상세 박스스코어 + Massey 5개 시스템 + 시드 쌍별 역대 승률 등 약 24개 피처 사용, 경기 유형별 가중치(토너먼트 5×, Secondary 2×) + 시즌 Recency 가중치 적용.

### 7.1 핵심 전략: Stage 1 Perfect Score Strategy

#### 핵심 인사이트
Stage 1은 **이미 플레이된 과거 시즌의 토너먼트 경기**를 평가 대상으로 삼음. 제출 파일에는 모든 가능한 팀 조합(519,144행)이 포함되지만, **실제 채점되는 것은 토너먼트에서 실제로 일어난 경기뿐**임.

#### 접근 방식
1. **제출 ID 파싱:** `{Season}_{T1}_{T2}` 형식에서 시즌, 팀1(작은 ID), 팀2(큰 ID) 추출
2. **결과 조회 (Lookup):** 해당 매치업이 과거 토너먼트(+ Secondary 토너먼트)에서 실제로 일어난 경기인지 확인
   - **YES →** 실제 결과에 따라 `0.999`(T1 승) 또는 `0.001`(T1 패) 부여
   - **NO →** ML 모델로 예측
3. **채점 원리:** Kaggle은 실제로 일어나지 않은 경기에 대해서는 채점 가중치가 0이므로, Lookup으로 맞힌 경기만 사실상 점수에 기여

#### 실행 결과
```
Total rows:                   519,144
Direct lookup (known result): 1,060   ← 거의 0 log-loss
Need ML prediction:           518,084
Lookup coverage:              0.2%
```
- Lookup 행 중: 534건 승리 (0.999), 526건 패배 (0.001)
- **기대 Lookup Log-Loss:** `−log(0.999) = 0.001001` per row → 전체 약 **0.00100** 수준

### 7.2 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | Elo 산출, 기본 팀 통계 (승률, 득실 마진 등) |
| `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` | 상세 박스스코어 (FG%, 3P%, FT%, 리바운드, 어시스트, 턴오버 등) |
| `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` | 결과 조회(Lookup) 테이블 + Elo 산출 + 학습 데이터 (가중치 5×) |
| `MNCAATourneyDetailedResults.csv` / `WNCAATourneyDetailedResults.csv` | 토너먼트 상세 결과 (읽기만 하고 직접적 피처 미사용) |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 번호, 시드 쌍별 역대 승률 |
| `MSecondaryTourneyCompactResults.csv` / `WSecondaryTourneyCompactResults.csv` | Lookup 확장 + 학습 데이터 (가중치 2×) |
| `MMasseyOrdinals.csv` | 5개 랭킹 시스템 (POM, SAG, WOL, MOR, DOK) — **남성 전용** |
| `MTeamConferences.csv` / `WTeamConferences.csv` | 컨퍼런스 코드 (카테고리형 피처) |
| `MTeams.csv` / `WTeams.csv` | 남/녀 팀 ID 구분용 |
| `SampleSubmissionStage1.csv` | 제출 양식 (519,144행) |

> 다른 노트북과의 차이점: `MTeamCoaches` **미사용**, Secondary 토너먼트를 Lookup + 학습 데이터 양쪽에 활용

### 7.3 전처리 및 피처 엔지니어링 (Feature Engineering)

> **총 ~24개 피처** = 팀 통계 차이(diff) 18개 + T1 경기수 + T2 경기수 + 시드 쌍별 승률 + 시즌 번호 (남성은 +6 Massey 피처)

#### A. 결과 조회 테이블 구축 (`build_result_lookup`)

- **범위:** NCAA 토너먼트 + Secondary 토너먼트 (NIT, CBI 등) 합산
- **키:** `(Season, min(WTeamID, LTeamID), max(WTeamID, LTeamID))`
- **값:** `1` (낮은 ID가 승리) 또는 `0` (높은 ID가 승리)
- **결과:** 남성 4,428건, 여성 2,623건 인덱싱 (1985~2025 시즌 커버)

#### B. Elo Rating (`compute_elo`) — 팀 피처: `Elo` (1개)

정규시즌 + 토너먼트 전체 경기를 시간순으로 합산하여 계산:

- **초기값:** `start = 1500`
- **시즌 간 평균 회귀 (Mean Reversion):**
  ```python
  cur[t] = elo.get((t, s-1), start) * (1 - revert) + start * revert   # revert = 0.33
  ```
  전년도 Elo를 67% 유지, 33%를 1500으로 회귀.
- **기대 승률:**
  ```python
  ea = 1 / (1 + 10^((cur[tb] - cur[ta]) / 400))
  ```
- **마진 승수 (Margin Multiplier):**
  ```python
  mult = log1p(|sa - sb|) × 2.2 / (0.001 + |cur[ta] - cur[tb]| / 400)
  ```
  점수 차이의 로그에 Elo 차이의 역수를 곱한 승수. 접전일수록(Elo 차이 작을수록) 승수가 커짐.
- **K-factor:** 기본 `k = 20`, 마진 승수 적용 후 `k2 = k × mult`
- **업데이트:** `cur[ta] += k2 × (1 − ea)`, `cur[tb] -= k2 × (1 − ea)`
- **홈 코트 어드밴티지:** 미적용 (Notebook 1, 3, 5, 6과 차이)

#### C. 시드 쌍별 역대 승률 (`seed_win_rates`) — 매치업 피처: `swr` (1개)

- 토너먼트 경기에서 승자/패자의 시드 번호를 매핑
- 시드 쌍 `(s1, s2)` 단위 (s1 < s2)로 역대 승률 집계:
  ```python
  wr = w / g   # w: 낮은 시드가 이긴 횟수, g: 총 대결 수
  ```
- 매치업 피처 생성 시 T1 시드와 T2 시드 기준 조회, 시드 역전 시 `1 - wr` 적용
- 해당 시드 쌍 이력 없으면 **기본값 `0.5`**

#### D. 팀 시즌 통계 (`team_stats`) — 팀 피처: 18개 (남성) / 12개 (여성)

정규시즌 + 토너먼트 + Secondary 토너먼트 경기를 합산하여 시즌별 팀 통계 산출:

**기본 집계 (CompactResults 기반, 6개):**

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `ng` | `count(Win)` | 총 경기 수 |
| `wp` | `mean(Win)` | 승률 |
| `pts` | `mean(Sc)` | 평균 득점 |
| `opp` | `mean(Opp)` | 평균 실점 |
| `mg` | `mean(Margin)` | 평균 점수 마진 |
| `smg` | `std(Margin)` | 마진 표준편차 (일관성 지표) |

**상세 박스스코어 (DetailedResults 기반, 9개):**

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `fg` | `FGM / FGA` | 필드골 성공률 |
| `fg3` | `FGM3 / FGA3` | 3점 성공률 |
| `ft` | `FTM / FTA` | 자유투 성공률 |
| `OR` | `sum(OR)` | 총 공격 리바운드 (평균 아닌 누적) |
| `DR` | `sum(DR)` | 총 수비 리바운드 |
| `Ast` | `sum(Ast)` | 총 어시스트 |
| `TO` | `sum(TO)` | 총 턴오버 |
| `Stl` | `sum(Stl)` | 총 스틸 |
| `Blk` | `sum(Blk)` | 총 블록 |

> **주의:** 리바운드, 어시스트 등은 경기당 평균이 아닌 **시즌 누적 합계**로 저장됨. 이는 경기 수(`ng`)와의 상호작용으로 모델이 간접적으로 경기당 수치를 학습하도록 설계된 것으로 보임.

**시드 번호 (`SeedNum`, 1개):**
- `re.findall(r'\d+', Seed)` 로 숫자 추출 (예: `'W01'` → `1`)
- 결측 시 `NaN` → `fillna(0)` 처리

**컨퍼런스 코드 (`ConfCode`, 1개):**
- `ConfAbbrev`를 `category` dtype으로 변환 후 `.cat.codes`로 정수 인코딩
- 결측 시 `fillna(0)`

**Massey Ordinals (남성 전용, 6개):**

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `POM` | 시즌 마지막 DayNum 기준 OrdinalRank | Pomeroy 랭킹 |
| `SAG` | 동일 | Sagarin 랭킹 |
| `WOL` | 동일 | Wolfe 랭킹 |
| `MOR` | 동일 | Moore 랭킹 |
| `DOK` | 동일 | Dokter Entropy 랭킹 |
| `AvgMassey` | `mean(POM, SAG, WOL, MOR, DOK)` | 5개 시스템 평균 랭킹 |

- 선별 기준: `GOOD_SYS = ['POM', 'SAG', 'WOL', 'MOR', 'DOK']` (5개 시스템만 사용)
- 결측값: 시스템별 전체 중앙값(`median`)으로 대체
- 여성 부문은 Massey 데이터 없음 → `mas_piv = None` → 6개 피처 미생성

#### E. 매치업 피처 조립 (`build_train` 내 동적 생성)

```python
diff = f1 - f2                                     # 팀 통계 차이 (18개)
feat = np.concatenate([diff, [f1[0], f2[0], swr_v, season]])
# f1[0] = T1의 ng (경기 수), f2[0] = T2의 ng, swr_v = 시드쌍 승률, season = 시즌 번호
```

**최종 피처 구성:**

| 피처 그룹 | 변수명 | 개수 | 설명 |
|-----------|--------|------|------|
| 차이 (diff) | `d_ng, d_wp, d_pts, d_opp, d_mg, d_smg, d_fg, d_fg3, d_ft, d_OR, d_DR, d_Ast, d_TO, d_Stl, d_Blk, d_SeedNum, d_ConfCode, d_Elo` | 18개 | T1 − T2 값 |
| Massey 차이 (남성) | `d_POM, d_SAG, d_WOL, d_MOR, d_DOK, d_AvgMassey` | 6개 | T1 − T2 Massey 랭킹 |
| 부가 피처 | `ng1` (T1 경기수), `ng2` (T2 경기수), `swr` (시드쌍 승률), `sn` (시즌) | 4개 | 매치업 메타 정보 |
| **합계** | | **남성 28개 / 여성 22개** | |

### 7.4 학습 데이터 구성

#### 학습 데이터 범위
- **시즌:** 2003~2025 (23개 시즌, 2020 제외)
- **경기 소스:** 정규시즌 + NCAA 토너먼트 + Secondary 토너먼트 **전부 포함**
- **남녀 분리:** 독립적으로 학습 데이터 구축

#### 경기 유형별 가중치 (Sample Weight)
```python
reg_s['wt']   = 1.0   # 정규시즌: 기본 가중치
tourn_s['wt'] = 5.0   # NCAA 토너먼트: 5배 가중
sec_s['wt']   = 2.0   # Secondary 토너먼트 (NIT 등): 2배 가중
```

#### 시즌 Recency 가중치 (선형 증가)
```python
rec_wt = 1.0 + 1.5 × (season - start) / (end - start)
# 2003 → 1.0, 2014 → 1.75, 2025 → 2.5
```
- 다른 노트북의 **지수 감쇠**(예: Notebook 1의 `0.60^(max-season)`)와 달리 **선형 증가** 방식
- 최종 가중치 = `경기유형 가중치 × 시즌 Recency 가중치`
  - 예: 2025 토너먼트 경기 = `5.0 × 2.5 = 12.5`

#### 라벨 설계
- 정준 순서: T1 = min(WTeamID, LTeamID), T2 = max
- `Target = 1` (T1 승리) 또는 `Target = 0` (T2 승리)
- **Symmetric Augmentation 미사용** — 원본 경기만 사용 (T1 < T2 단일 방향)

#### 데이터 규모
```
Men:   121,612행
Women: 117,949행
```
> 정규시즌 경기를 포함하므로 토너먼트만 사용하는 Notebook 1(~2,200행)이나 Notebook 3(~2,400행) 대비 **50~55배** 대규모

### 7.5 모델 학습 (5-fold Stratified CV 앙상블)

> **하이퍼파라미터 튜닝 없음** — 모든 파라미터 하드코딩. Optuna 미사용.

#### 3개 모델 공통 설정

| 파라미터 | XGBoost | LightGBM | CatBoost |
|----------|---------|----------|----------|
| `n_estimators / iterations` | 800 | 800 | 800 |
| `max_depth / depth` | 5 | 5 | 5 |
| `learning_rate` | 0.015 | 0.015 | 0.015 |
| `subsample` | 0.8 | 0.8 | — |
| `colsample_bytree` | 0.8 | 0.8 | — |
| `reg_alpha` | 0.3 | 0.3 | — |
| `reg_lambda / l2_leaf_reg` | 2.0 | 2.0 | 4.0 |
| `min_child_weight / min_child_samples` | 5 | 25 | — |
| `random_state / random_seed` | 42 | 42 | 42 |

> 3개 모델이 거의 동일한 구조(depth=5, lr=0.015, 800 rounds)로 설정됨 — 다양성보다 안정성 중시.

#### CV 전략
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 각 fold에서 3개 모델 독립 학습 → 5개 fold × 3개 모델 = **15개 모델**
- **Sample Weight 적용:** `sample_weight=wtr` (경기 유형 × 시즌 Recency)
- **Early Stopping 미사용** — 고정 800 라운드 (eval_set은 설정되나 early_stopping_rounds 미지정)

#### OOF 앙상블 방식 (학습 시)
```python
oof[vai] = (px + pl + pc) / 3   # 3개 모델 단순 1/3 평균
```

#### OOF 성능 결과

| 지표 | Men's | Women's |
|------|-------|---------|
| **OOF LogLoss** | 0.48982 | 0.46236 |
| **OOF Brier Score** | 0.16282 | 0.15237 |

### 7.6 예측 및 제출

#### 예측 파이프라인 (2단계 하이브리드)

**Step 1 — Lookup (실제 발생 경기):**
- 제출 ID의 `(Season, T1, T2)` 가 `m_lookup` / `w_lookup` 에 존재하면 실제 결과 부여
- 클리핑: `[0.001, 0.999]` — 극단적으로 좁은 범위 (Log-Loss ≈ 0.001/row)

**Step 2 — ML 예측 (미발생 경기):**
- 성별 분리: `T1 ∈ MTeams` → 남성 모델, otherwise → 여성 모델
- 2026 시즌 처리: `ref = min(season, 2025)` — 2026 시즌은 2025 데이터로 대체
- 5-fold 모델의 평균 예측:
  ```python
  px = mean([m.predict_proba(Xp)[:,1] for m in xms])   # XGBoost 5-fold 평균
  pl = mean([m.predict_proba(Xp)[:,1] for m in lms])   # LightGBM 5-fold 평균
  pc = mean([m.predict_proba(Xp)[:,1] for m in cms])   # CatBoost 5-fold 평균
  ```
- **가중 앙상블:**
  ```python
  preds = 0.35 × px + 0.35 × pl + 0.30 × pc   # XGB+LGB 각 35%, CatBoost 30%
  ```
- ML 클리핑: `[0.05, 0.95]` — Lookup보다 넓은 범위

**Step 3 — 결합:**
- Lookup 행은 그대로 유지 (0.999 / 0.001)
- ML 행은 예측값 사용, 매핑 실패 시 `0.5` (기본값)

#### 제출 통계
```
Overall Pred range:  [0.001, 0.999]
ML Pred range:       [0.050, 0.950]
Lookup distribution: 534 wins (0.999), 526 losses (0.001)
Mean: 0.4952 | Std: 0.3233
25%: 0.1810 | 50%: 0.4872 | 75%: 0.8112
```

### 7.7 기대 성능 분석

노트북 내 Sanity Check 결과:
```
Expected log-loss per lookup row: 0.001001
Total lookup rows:                1,060
Expected total loss contribution: 1.060530
Expected mean log-loss:           0.001001
```
→ Stage 1에서 실제 채점되는 행이 모두 Lookup 행이라면 **Log-Loss ≈ 0.00100** 달성 가능.

### 7.8 주요 설계 특징 및 비교

| 항목 | Notebook 7 (본 노트북) | 다른 노트북 |
|------|----------------------|-------------|
| **핵심 전략** | **하이브리드 (Lookup + ML)** | 순수 ML 예측 |
| **Stage 1 최적화** | 과거 결과 직접 조회 → 거의 0 loss | ML 모델로 확률 예측 |
| **피처 수** | 남성 28개 / 여성 22개 | 13~344개 |
| **학습 데이터** | 정규시즌 + 토너먼트 + Secondary (12만행+) | 토너먼트만(~2K) ~ 전체(~39만행) |
| **가중치 방식** | 경기 유형별 (토너먼트 5×) + 시즌 선형 | 시즌 지수 감쇠 또는 없음 |
| **앙상블** | 35/35/30 가중 평균 | 단순 평균, Stacking, MoE 등 |
| **CV 전략** | StratifiedKFold (무작위) | Temporal 또는 StratifiedKFold |
| **보정** | 없음 (raw 앙상블) | Isotonic, Logit Calibration |
| **Massey 시스템** | POM, SAG, WOL, MOR, DOK (5개) | POM, SAG, COL, DOL, MOR, WLK, RTH (7개) 등 |
| **Elo 홈 어드밴티지** | **미적용** | 75~100 포인트 보정 |
| **Elo 시즌 회귀** | 0.33 (33%) | 0.25~0.30 |
| **클리핑 전략** | Lookup: [0.001, 0.999] / ML: [0.05, 0.95] — **이중 클리핑** | 단일 [0.02, 0.98] ~ [0.025, 0.975] |

### 7.9 종합 코멘트 (Review)

**장점:**
- Stage 1 평가 구조를 정확히 이해하고, Lookup 전략으로 **이론적 최소 Log-Loss (~0.001)** 달성 가능
- Secondary 토너먼트 결과까지 포함하여 Lookup 커버리지 극대화 (남성 4,428건, 여성 2,623건)
- 경기 유형별 가중치(토너먼트 5×)로 ML 모델이 토너먼트 패턴에 집중하도록 유도
- 멀티 소스 피처 (Elo + 상세 박스스코어 + Massey + 시드 쌍별 승률 + 컨퍼런스) 결합

**한계점:**
- **Stage 2에서는 무력화:** 2026 시즌 토너먼트는 미래 데이터이므로 Lookup 불가 → 순수 ML 성능에 의존
- ML 모델의 OOF 성능(LogLoss 0.49, Brier 0.16)은 **다른 노트북 대비 낮은 수준** — 정규시즌 경기의 노이즈가 학습을 방해할 가능성
- 상세 박스스코어 피처(OR, DR, Ast 등)가 **누적 합계**로 저장되어 경기 수가 크게 다른 팀 간 비교 시 왜곡 가능
- Early Stopping 미사용, 하이퍼파라미터 튜닝 없음
- Elo 산출 시 홈 코트 어드밴티지 미적용
- 여성 부문 Massey 피처 부재 (6개 피처 탈락)
- **Lookup 전략은 대회 규정상 "기존 결과를 제출에 하드코딩" 하는 방식이므로, 순수 예측 능력 관점에서는 의미가 제한적**

---

## 8. march-machine-learning-mania-2026-notebook.ipynb
**주요 특징:** 단일 셀(Single Cell)로 구성된 **초간결 파이프라인**. 남녀 데이터를 합쳐서 단일 모델로 학습하며, 4개의 차이 피처(승률, 득실 마진, 시드, Massey 랭킹)만으로 XGBoost + LightGBM 단순 1/2 평균 앙상블. CV도 보정도 없는 순수 베이스라인.

### 8.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` | 읽기만 함 (실제 코드에서 직접적 사용 없음) |
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | 승률, 득실 마진 산출 |
| `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` | 학습 데이터 (토너먼트 경기) |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 번호 추출 |
| `MMasseyOrdinals.csv` | DayNum=133의 전체 시스템 평균 랭킹 — **남성 전용** |
| `SampleSubmissionStage1.csv` | 제출 양식 |

> **특이사항:** `DetailedResults`를 import하지만 **피처에 사용하지 않음**. 6개 노트북 중 유일하게 **남녀 데이터를 합쳐서(concat) 단일 모델**로 학습.

### 8.2 전처리 및 피처 엔지니어링 (Feature Engineering)

> **총 4개 피처** — 6개 노트북 중 가장 단순

#### A. 남녀 합산 (Gender-Agnostic)
```python
reg    = pd.concat([M_reg, W_reg])       # DetailedResults (미활용)
compact= pd.concat([M_comp, W_comp])     # CompactResults
tour   = pd.concat([M_tour, W_tour])     # Tournament
seeds  = pd.concat([M_seeds, W_seeds])   # Seeds
```
- 남녀를 구분하지 않고 단일 데이터셋으로 합산

#### B. 승률 (`WinRate`) — 팀 피처 1개
```python
Wins = groupby(['Season','WTeamID']).size()
Loss = groupby(['Season','LTeamID']).size()
WinRate = Wins / (Wins + Loss)
```
- `CompactResults` 기반, 정규시즌 전체 경기

#### C. 득실 마진 강도 (`ScoreStrength`) — 팀 피처 1개
```python
ScoreDiff = WScore − LScore
ScoreW = groupby(['Season','WTeamID'])['ScoreDiff'].mean()   # 승리 시 평균 마진
ScoreL = groupby(['Season','LTeamID'])['ScoreDiff'].mean()   # 패배 시 평균 마진
ScoreStrength = ScoreW − ScoreL
```
- 승리 시 평균 마진에서 패배 시 평균 마진을 빼는 방식 — **다른 노트북의 단순 마진(mean(ScoreFor − ScoreAgainst))과 다른 독특한 설계**

#### D. 시드 번호 (`SeedNum`) — 팀 피처 1개
```python
SeedNum = Seed.str[1:3].astype(int)   # 예: 'W01' → 1
```

#### E. Massey 랭킹 (`MasseyRank`) — 팀 피처 1개
```python
massey = massey[massey['RankingDayNum'] == 133]   # DayNum=133만 필터
MasseyRank = groupby(['Season','TeamID'])['OrdinalRank'].mean()
```
- **DayNum=133 고정** (시즌 마지막 날), **전체 시스템 평균** (선별 없음)
- 남성 전용 → 여성은 `NaN` → `fillna(0)`

#### F. 매치업 차이 피처 (4개)

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `WR_diff` | `T1_WR − T2_WR` | 승률 차이 |
| `Score_diff` | `T1_Score − T2_Score` | 득실 마진 강도 차이 |
| `Seed_diff` | `T2_Seed − T1_Seed` | 시드 차이 (T2 − T1, **부호 반전 주의**) |
| `Rank_diff` | `T2_Rank − T1_Rank` | Massey 랭킹 차이 (T2 − T1, 높을수록 T1 유리) |

> **주의:** `Seed_diff`와 `Rank_diff`는 `T2 − T1`로 계산됨 (다른 노트북은 보통 `T1 − T2`). 이는 "높은 값 = T1에 유리"하도록 부호를 맞춘 것 (시드와 랭킹은 낮을수록 강함).

### 8.3 학습 데이터 구성

#### Data Augmentation (승/패 대칭)
```python
rows.append([Season, WTeamID, LTeamID, 1])   # 승자 = Team1
rows.append([Season, LTeamID, WTeamID, 0])   # 패자 = Team1
```
- 각 토너먼트 경기에서 승/패 양방향 행 생성 → **2배 증강**
- **남녀 합산:** `tour = concat([M_tour, W_tour])` → 단일 학습셋
- 결과: **8,604행** (남녀 토너먼트 전체 × 2)
- **정준 순서(T1 < T2) 미사용** — 승자를 Team1으로 놓는 방식

#### 결측 처리
- 전체 피처: `fillna(0)` (결측값을 0으로 대체)

### 8.4 모델 학습 (전체 데이터, CV 없음)

> **교차 검증 없음** — 전체 학습 데이터로 단일 모델 학습

**XGBoost:**
| 파라미터 | 값 |
|----------|-----|
| `n_estimators` | 1500 |
| `learning_rate` | 0.01 |
| `max_depth` | 4 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `tree_method` | 'hist' |

**LightGBM:**
| 파라미터 | 값 |
|----------|-----|
| `n_estimators` | 1500 |
| `learning_rate` | 0.01 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

> Early Stopping 없이 고정 1500 라운드 학습. 검증 없음.

### 8.5 예측 및 제출

```python
pred1 = xgb.predict_proba(sub[features].fillna(0))[:,1]
pred2 = lgb.predict_proba(sub[features].fillna(0))[:,1]
sub["Pred"] = (pred1 + pred2) / 2   # 단순 1/2 평균
```
- **앙상블:** XGBoost + LightGBM 단순 1/2 평균
- **클리핑 없음** — 확률값을 raw 출력 그대로 사용
- **성별 분리 없음** — 단일 모델로 남녀 모두 예측

### 8.6 실행 출력 분석
```
Number of data points in the train set: 8604, number of used features: 4
```
- LightGBM이 인식한 학습 데이터: 8,604행, 4개 피처
- 기본 양성 비율: `pavg=0.500000` → 완벽한 클래스 균형 (대칭 증강 결과)

### 8.7 주요 설계 특징 및 비교

| 항목 | Notebook 8 (본 노트북) | 다른 노트북 |
|------|----------------------|-------------|
| **코드 구조** | **단일 셀** | 다중 셀/모듈화 |
| **피처 수** | **4개** (최소) | 13~344개 |
| **남녀 처리** | **합산 (단일 모델)** | 분리 (별도 모델) |
| **CV** | **없음** | StratifiedKFold / Temporal |
| **앙상블** | XGB + LGB 1/2 평균 | 3~11개 모델 |
| **보정** | **없음** | Isotonic, Logit |
| **클리핑** | **없음** | [0.001, 0.999] ~ [0.025, 0.975] |
| **Augmentation** | 승/패 대칭 (T1=승자) | 정준 순서 기반 |
| **Massey 처리** | DayNum=133, 전체 시스템 평균 | 선별 시스템, 백분위 변환 등 |
| **Elo** | **미사용** | 거의 모든 노트북에서 사용 |

### 8.8 종합 코멘트 (Review)

**장점:**
- 극도로 간결하여 이해 및 실행이 용이 (단일 셀, 60초 내 실행)
- 남녀 합산으로 데이터 규모 증가 (8,604행)
- `ScoreStrength` 피처 설계가 독특 (승리 마진 − 패배 마진)

**한계점:**
- **피처 4개로 정보 부족** — Elo, 상세 박스스코어, 컨퍼런스 등 전혀 미사용
- **교차 검증 없음** → 과적합 위험 감지 불가
- **클리핑 없음** → 극단적 확률값이 Brier/LogLoss 악화 가능
- **남녀 합산 학습**이 성별 간 분포 차이(업셋 비율 등)를 무시
- 정준 순서(T1 < T2)를 사용하지 않아 모델이 Team1/Team2 순서에 따른 편향 학습 가능
- LightGBM에서 `Seed_diff` 방향(T2−T1)이 다른 피처(T1−T2)와 반대 → 해석 혼란
- Massey 랭킹이 여성에게 `NaN → 0`으로 처리되어 남녀 합산 시 여성 예측에 노이즈 유발
- **출력 셀에 성능 지표 없음** — Brier/LogLoss 미측정

---

## 9. march-machine-learning-mania-2026.ipynb
**주요 특징:** 체계적인 학습 파이프라인 — DetailedResults 기반 효율성 지표(OffEff, DefEff, NetRating, Possessions) + Elo Rating + 시드 → **Time Series CV**(시간 순차 검증) → LogReg(Baseline) + LightGBM + XGBoost → Grid Search 최적 가중치 앙상블 → Isotonic Calibration. 남녀 **완전 분리** 파이프라인으로, 노트북 내 마크다운이 각 단계의 의사결정 과정을 상세히 기록.

### 9.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | (읽기, 직접 활용 안 함) |
| `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` | 팀-게임 상세 통계 산출 (FG%, OR, DR, Ast, TO 등) |
| `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` | 학습 데이터 (토너먼트 매치업) |
| `MNCAATourneyDetailedResults.csv` / `WNCAATourneyDetailedResults.csv` | (읽기, 직접 활용 안 함) |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 번호 |
| `MMasseyOrdinals.csv` | (읽기만, 피처에 미사용) |
| `SampleSubmissionStage1.csv` | 제출 양식 |

> **핵심 데이터:** `DetailedResults`로부터 팀별 경기 통계를 팀 관점(Team-Centric)으로 변환. **DayNum ≤ 132 필터** 적용으로 토너먼트 전 데이터만 사용 (Leakage 방지).

### 9.2 전처리 및 피처 엔지니어링 (Feature Engineering)

> **남성: 8개 차이 피처, 여성: 7개 차이 피처** (Seed 피처가 여성에서 제외)

#### A. 데이터 필터링 (Leakage 방지)
```python
MRegularDetailed = MRegularDetailed[MRegularDetailed["DayNum"] <= 132]
WRegularDetailed = WRegularDetailed[WRegularDetailed["DayNum"] <= 132]
```
- 정규시즌만 사용하여 토너먼트 결과의 역류 차단

#### B. 팀 관점 변환 (`build_team_game_df`)
각 경기를 승자/패자 관점에서 2행으로 분리하여 팀-중심(Team-Centric) 데이터프레임으로 변환:
- 승자 행: `TeamID=WTeamID`, `PointsFor=WScore`, `PointsAgainst=LScore`, `Win=1`
- 패자 행: `TeamID=LTeamID`, `PointsFor=LScore`, `PointsAgainst=WScore`, `Win=0`
- 각 행에 FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF 포함

#### C. 시즌별 팀 통계 (`build_season_features`) — 팀 피처 6개

`groupby(['Season','TeamID'])`로 시즌 전체 누적 합계 후 파생 변수 산출:

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `WinPct` | `Win / Games` | 승률 |
| `AvgMargin` | `(PointsFor − PointsAgainst) / Games` | 평균 득실 마진 |
| `Possessions` | `FGA − OR + TO + 0.475 × FTA` | 시즌 총 포제션 (0.475 계수) |
| `OffEff` | `PointsFor / Possessions × 100` | 공격 효율 (100포제션당 득점) |
| `DefEff` | `PointsAgainst / Possessions × 100` | 수비 효율 (100포제션당 실점, 낮을수록 좋음) |
| `NetRating` | `OffEff − DefEff` | 순 효율 |

#### D. Elo Rating (`compute_elo`) — 팀 피처: `EloRating` (1개)

팀-게임 데이터프레임에서 시간순으로 계산:

- **초기값:** `base_elo = 1500`
- **시즌 간 평균 회귀:** `new = prev × (1 − regression) + base × regression`, `regression = 0.25`
  - 전년도 Elo의 **75% 유지**, 25%를 1500으로 회귀
- **기대 승률:** `exp = 1 / (1 + 10^((opp_elo − team_elo) / 400))`
- **마진 승수:**
  ```python
  margin_mult = log(margin + 1) × (2.2 / ((team_elo − opp_elo) × 0.001 + 2.2))
  ```
- **K-factor:** `k = 20`, 마진 승수 적용
- **업데이트:** `new_elo = team_elo + k × margin_mult × (win − exp)`
- **홈 코트 어드밴티지:** 코드에 `home_adv=65` 매개변수가 있으나 **실제 계산에 미사용** (정규시즌 DetailedResults에 WLoc 없음)
- **Pre-Tournament Elo:** 시즌 마지막 경기의 `EloPre` 값을 시즌별 팀 Elo로 추출

> 남녀 각각 **독립적으로** Elo 계산

#### E. 시드 (`Seed`) — 남성만

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Seed_T1` | `Seed.str[1:3].astype(int)` | 시드 번호 (1~16) |
| `Seed_T2` | 동일 | |
| `Seed_Diff` | `Seed_T1 − Seed_T2` | 시드 차이 |

- **남성에만 적용** → 여성에는 Seed 피처 미포함 (7개 피처)
- 결측 시 `fillna(16)` (최약 시드로 대체)

#### F. 최종 매치업 차이 피처

**남성 (8개):**
```python
feature_cols = ["WinPct", "AvgMargin", "OffEff", "DefEff",
                "NetRating", "Possessions", "EloRating", "Seed"]
# 각 col에 대해: {col}_Diff = {col}_T1 − {col}_T2
```

**여성 (7개):** Seed 제외
```python
feature_cols = ["WinPct", "AvgMargin", "OffEff", "DefEff",
                "NetRating", "Possessions", "EloRating"]
```

### 9.3 학습 데이터 구성

#### 라벨 설계
- **정준 순서:** `Team1 = min(WTeamID, LTeamID)`, `Team2 = max`
- `Target = 1` (Team1 = 낮은 ID가 승리) 또는 `0`
- **Augmentation 없음** — 토너먼트 경기 원본만 사용

#### 데이터 범위
- **남성:** Season ≥ 2003 (DetailedResults 시작점) → NaN 행 제거
- **여성:** Season ≥ 2010 (여성 DetailedResults 시작점)
- **남녀 독립 데이터셋**

### 9.4 교차 검증 (Time Series CV)

> **시간 기반 순차 검증** — 미래 데이터 누수 완전 차단

```python
for i in range(3, len(seasons)):
    train_seasons = seasons[:i]   # 과거 시즌들
    val_season = seasons[i]       # 현재 시즌 1개
```
- **남성:** 20개 fold (2003~2005 train → 2006 val, ..., 2003~2024 train → 2025 val)
- **여성:** 13개 fold (2010~2012 train → 2013 val, ..., 2010~2024 train → 2025 val)
- 평가 지표: **Brier Score**

### 9.5 모델 학습

#### Baseline: Logistic Regression
- `StandardScaler` → `LogisticRegression(max_iter=1000)`
- 각 fold별 스케일링 (fit on train, transform on val)

#### LightGBM
```python
lgb_params = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": 0.03, "num_leaves": 31,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
              "min_data_in_leaf": 30, "verbosity": -1}
```
- `num_boost_round=1000`, `early_stopping(100)` 적용

#### XGBoost
```python
xgb_params = {"objective": "binary:logistic", "eval_metric": "logloss",
              "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 5, "seed": 42}
```
- `n_estimators=2000`, `early_stopping_rounds=100` 적용

### 9.6 OOF 성능 결과 (노트북 마크다운 기록)

| 모델 | Men's Brier | Women's Brier |
|------|-------------|---------------|
| Logistic Regression | 0.18971 | **0.14331** |
| LightGBM | **0.18836** | 0.14729 |
| XGBoost | 0.18841 | 0.17903 |

> **남성:** LightGBM ≈ XGBoost > Logistic (트리 모델 소폭 우위)
> **여성:** Logistic > LightGBM > XGBoost (선형 모델이 확실히 우수 — 여성 데이터의 선형 분리 가능성이 높고 트리가 과적합)

### 9.7 앙상블 (Grid Search 최적 가중치)

#### 가중치 탐색 (0.05 간격 그리드 탐색)
```python
for w1 in np.arange(0.0, 1.01, 0.05):       # Logistic
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):  # LightGBM
        w3 = 1 - w1 - w2                        # XGBoost
```

#### 최적 가중치 결과 (노트북 마크다운 추론)
- **남성:** Logistic 45% + LightGBM 55% (XGBoost 0%) → Brier 0.1855
- **여성:** Logistic 위주 (XGBoost 무시) → Brier 0.14254

### 9.8 Isotonic Calibration

```python
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(ensemble_oof[valid_mask], y_true[valid_mask])
calibrated = iso.predict(ensemble_oof[valid_mask])
```
- OOF 앙상블 예측값을 단조 변환으로 보정

#### 보정 후 성능 (노트북 마크다운 기록)

| 단계 | Men's Brier | Women's Brier |
|------|-------------|---------------|
| Best Single Model | 0.18836 | 0.14331 |
| 최적 가중 앙상블 | 0.18550 | 0.14254 |
| **Isotonic Calibration** | **0.18097** | **0.13657** |

> 남성: 앙상블 → 보정으로 Brier **0.18836 → 0.18097** (Δ = −0.00739)
> 여성: 앙상블 → 보정으로 Brier **0.14331 → 0.13657** (Δ = −0.00674)

### 9.9 예측 및 제출

#### 최종 학습
- **Logistic Regression:** 전체 데이터로 `StandardScaler` + `LogisticRegression(max_iter=1000)` 학습
- **LightGBM:** 전체 데이터로 `n_estimators=1000, lr=0.05` 학습 (CV보다 높은 lr)
- **제출에는 LightGBM 단독 예측 사용** (앙상블이 아닌, `lgb_full_men/women.predict_proba()` 직접 출력)

#### 성별 분리
- `Team1 < 3000` → 남성 모델, `Team1 >= 3000` → 여성 모델

#### 결측 처리
- 시드 결측: `fillna(16)` (최약 시드)
- 기타 피처: `fillna(0)`

#### 클리핑
- **없음** — raw LightGBM 예측값 그대로 사용

### 9.10 주요 설계 특징 및 비교

| 항목 | Notebook 9 (본 노트북) | 다른 노트북 |
|------|----------------------|-------------|
| **피처 수** | 남성 8 / 여성 7 | 4~344개 |
| **CV 전략** | **Time Series (시간순)** | StratifiedKFold / Temporal |
| **앙상블 가중치** | **Grid Search (OOF 기반)** | 수동, Inverse-Brier, SLSQP 등 |
| **보정** | **Isotonic Calibration** | 없음 ~ Logit Calibration |
| **Elo 시즌 회귀** | 0.25 (25%) | 0.25~0.33 |
| **포제션 계수** | **0.475** | 0.44 ~ 0.475 |
| **효율성 지표** | OffEff, DefEff, NetRating | 일부 노트북만 |
| **클리핑** | **없음** | [0.001, 0.999] ~ [0.05, 0.95] |
| **제출 모델** | **LightGBM 단독** (앙상블 아님) | 앙상블 |
| **남녀 처리** | **완전 독립** | 합산 ~ 독립 |
| **마크다운 분석** | **매우 상세** (의사결정 기록) | 코드 위주 |

### 9.11 종합 코멘트 (Review)

**장점:**
- **Time Series CV**로 미래 데이터 누수 완전 차단 — 현실적인 검증 전략
- **DayNum ≤ 132 필터**로 정규시즌 데이터만 사용 (추가 Leakage 방지)
- **Grid Search 가중치 탐색**으로 모델 간 최적 배분 자동화
- **Isotonic Calibration**으로 확률 보정 효과 실증 (Men −0.007, Women −0.007)
- 노트북 마크다운에 **각 모델의 장단점 분석**(여성에서 트리 모델이 과적합하는 이유 등) 기록
- DetailedResults 기반 효율성 지표(OffEff, DefEff)로 포제션 품질 반영

**한계점:**
- **제출 시 앙상블+보정을 적용하지 않고 LightGBM 단독 사용** — OOF에서 확인된 이점이 제출에 반영 안 됨
- 클리핑 없음 → 극단적 확률값으로 Brier Score 불이익 가능
- 피처 수(7~8개)는 여전히 적음 — Massey, 컨퍼런스, 코치 등 외부 데이터 미활용
- `MMasseyOrdinals`를 읽지만 **피처로 사용하지 않음** (남성에만 Seed 피처 추가)
- Elo의 `home_adv=65` 파라미터가 코드에 있으나 실제 적용되지 않음
- **여성 시드 미사용** (7개 vs 남성 8개): 시드가 강력한 예측 변수임에도 불구
- LightGBM 최종 학습 시 `lr=0.05`로 CV(0.03)보다 높은 학습률 사용 — 과적합 위험

---

## 10. march-mania-2026.ipynb
**주요 특징:** 단일 셀(Single Cell) 풀 파이프라인이지만 **구조적 완성도가 높음**. Recency 가중 평균 기반 팀 통계(OffEff, DefEff 등 8개) + SoS + Massey 5개 시스템 + Elo → **LightGBM 5-fold CV** + **Elo 로지스틱 모델**(Scipy 최적 스케일 탐색) → **각각 독립적 Isotonic Calibration** → **자동 최적 가중치 앙상블**(LGB 55% + Elo 45%). 남녀 합산 학습하되 `Gender` 플래그로 구분. OOF Brier **0.1682**.

### 10.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | SoS 산출 + Elo 산출 |
| `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` | Recency 가중 팀 통계 산출 (OffEff, DefEff 등 8개) |
| `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` | 학습 데이터 (토너먼트 매치업, 2003~) |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 번호 |
| `MMasseyOrdinals.csv` | POM, SAG, MOR, DOL, RPI 선별 (남성 전용) |
| `SampleSubmissionStage1.csv` | 제출 양식 (519,144행) |

> Massey 없는 여성은 `NetEff` 시즌 내 순위로 대체 (`rank(ascending=False)`)

### 10.2 전처리 및 피처 엔지니어링 (Feature Engineering)

> **총 13개 피처** = 10개 차이(diff) + Seed_diff + AbsSeed_diff + Gender

#### A. Recency 가중 팀 통계 (`compute_team_stats`) — 팀 피처: 8개

**Recency 가중치 (시즌 내 시간 기반):**
```python
RW = 0.5 + (DayNum − day_min) / (day_max − day_min + 1)
# 시즌 초반 → 0.5, 시즌 말 → ~1.5
```
- 각 경기의 시즌 내 상대적 위치에 따라 **선형 증가 가중치** 적용
- 시즌 후반 경기에 더 높은 비중을 부여 (최근 폼 반영)

**경기별 산출 지표 (Winner/Loser 양측 관점 분리):**

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `WinPct` | `weighted_mean(Win)` | 가중 승률 |
| `OffEff` | `pts / poss × 100` | 공격 효율 (100포제션당) |
| `DefEff` | `opts / opp_poss × 100` | 수비 효율 (100포제션당, 낮을수록 좋음) |
| `NetEff` | `OffEff − DefEff` | 순 효율 |
| `Pace` | `(poss + opp_poss) / 2` | 경기 속도 |
| `TS_pct` | `pts / (2 × (FGA + 0.44 × FTA))` | True Shooting % |
| `Reb_pct` | `(OR + DR) / (OR + DR + OppOR + OppDR)` | 리바운드 점유율 |
| `TO_pct` | `TO / poss` | 턴오버율 (낮을수록 좋음) |

- **포제션:** `Poss = FGA − OR + TO + 0.475 × FTA` (0.475 계수)
- **집계:** `groupby(['Season','TeamID'])`로 **가중 평균** (`np.average(values, weights=RW)`)

> 다른 노트북의 단순 `mean()`과 달리, **시즌 후반 경기에 비중을 더 준 가중 평균** 사용

#### B. Strength of Schedule (`compute_sos`) — 팀 피처: `SoS` (1개)

```python
SoS = mean(상대팀의 NetEff)   # 정규시즌 모든 상대의 순 효율 평균
```
- `CompactResults` 기반, Winner/Loser 양쪽 관점으로 상대팀 매핑
- 상대 NetEff가 없으면 `0.0` 대체

#### C. Massey Ordinals (`get_massey`) — 팀 피처: `MasseyRank` (1개)

**남성:**
- **선별 시스템:** `['POM', 'SAG', 'MOR', 'DOL', 'RPI']` (5개, 존재하는 것만)
- **필터:** `RankingDayNum ≤ 133`, 각 시스템별 마지막 DayNum만
- **집계:** `OrdinalRank`의 5개 시스템 **평균**

**여성 (Massey 미존재 대체):**
```python
w_stats['MasseyRank'] = w_stats.groupby('Season')['NetEff'].rank(ascending=False)
```
- NetEff 시즌 내 순위를 Massey 대체 변수로 활용 (1위 = 가장 강함)

#### D. Elo Rating (`compute_elo`) — 팀 피처: `Elo` (1개)

**정규시즌만** 사용하여 계산 (토너먼트 미포함):

- **초기값:** `DEFAULT = 1500.0`
- **시즌 간 평균 회귀:** `elo[tid] = elo[tid] × (1 − revert) + 1500 × revert`, `revert = 0.33`
  - **모든 팀을 매 시즌 초에 일괄 회귀** (시즌별 groupby 순회 시 수행)
- **기대 승률:** `exp_w = 1 / (1 + 10^((el − ew) / 400))`
- **마진 승수:**
  ```python
  mov = log(|margin| + 1) × (2.2 / (|ew − el| × 0.001 + 2.2))
  ```
- **K-factor:** `K = 20 × mov`
- **업데이트:** `elo[wt] += k × (1 − exp_w)`, `elo[lt] += k × (0 − (1 − exp_w))`
- **홈 코트 어드밴티지:** 미적용

> 남녀 각각 독립 계산, 시즌별 최종 Elo를 `elo_history`에 저장

#### E. 매치업 피처 조립 (`build_matchup_features`)

**차이 피처 (10개):** 부호 방향 주의
```python
LOWER_BETTER = {'DefEff', 'TO_pct', 'MasseyRank'}
# LOWER_BETTER: diff = T2 − T1 (높을수록 T1 유리)
# 나머지:        diff = T1 − T2 (높을수록 T1 유리)
```

| 변수명 | 산출 방향 | 설명 |
|--------|-----------|------|
| `Elo_diff` | T1 − T2 | Elo 차이 |
| `OffEff_diff` | T1 − T2 | 공격 효율 차이 |
| `DefEff_diff` | T2 − T1 | 수비 효율 차이 (T1이 낮을수록 유리) |
| `NetEff_diff` | T1 − T2 | 순 효율 차이 |
| `Pace_diff` | T1 − T2 | 페이스 차이 |
| `TS_pct_diff` | T1 − T2 | True Shooting 차이 |
| `Reb_pct_diff` | T1 − T2 | 리바운드율 차이 |
| `TO_pct_diff` | T2 − T1 | 턴오버율 차이 (T1이 낮을수록 유리) |
| `SoS_diff` | T1 − T2 | SoS 차이 |
| `MasseyRank_diff` | T2 − T1 | Massey 랭킹 차이 (T1이 순위 높을수록 유리) |

**시드 피처 (2개):**
| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Seed_diff` | `seed2 − seed1` | 시드 차이 (T1이 낮을수록 유리) |
| `AbsSeed_diff` | `|seed1 − seed2|` | 시드 격차 절대값 (매치업 난이도) |

**메타 피처 (1개):**
| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `Gender` | `0` (남성) / `1` (여성) | 성별 구분 플래그 |

- 시드 결측: **기본값 `8`** (중앙 시드)
- Elo 결측: **기본값 `1500.0`** (초기값)

### 10.3 학습 데이터 구성

- **범위:** Season ≥ 2003 (DetailedResults 시작점)
- **남녀 합산:** `all_matchups = concat([m_matchups, w_matchups])`
- **라벨:** 정준 순서 T1 < T2, `Target = 1` (T1 승리) 또는 `0`
- **Data Augmentation 없음** — 원본 토너먼트 경기만 사용
- **데이터 규모:** 남성 1,449행 + 여성 961행 = **2,410행**
- **양성 비율:** 0.502 (T1 < T2이므로 거의 균형)

### 10.4 모델 학습

#### Model 1: LightGBM (5-fold Stratified CV)

```python
lgb_params = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.02, "n_estimators": 1000, "num_leaves": 31,
    "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.5, "random_state": 42
}
```
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `early_stopping(50)` 적용
- **5개 fold 모델 모두 보존** (예측 시 5-fold 평균)

**Fold별 성능:**

| Fold | Best Iteration | Brier Score |
|------|---------------|-------------|
| 1 | 155 | 0.1732 |
| 2 | 101 | 0.1772 |
| 3 | 134 | 0.1767 |
| 4 | 131 | 0.1711 |
| 5 | 195 | 0.1708 |
| **OOF** | — | **0.1738** |

#### Model 2: Elo 로지스틱 모델 (Scipy 최적화)

```python
oof_elo = expit((Elo1 − Elo2) / best_scale)
# best_scale = scipy.optimize.minimize_scalar(Brier_Score, bounds=(50, 1000))
```
- **expit** = 시그모이드 함수: `1 / (1 + exp(−x))`
- **스케일 파라미터 최적화:** Brier Score를 최소화하는 `scale` 값을 `minimize_scalar`로 자동 탐색
  - 탐색 범위: `[50, 1000]`, 방법: `bounded`
  - 결과: **best_scale = 122.4**
- **OOF Elo Brier:** 0.1753

> 기존 Elo 승률 공식의 `400` 대신 `122.4`를 사용 — 더 넓은 확률 분포를 생성하여 "과신(overconfidence)"을 줄이고 Brier에 최적화

### 10.5 확률 보정 (Dual Isotonic Calibration)

```python
ir_lgb = IsotonicRegression(out_of_bounds='clip').fit(oof_lgb, y)
ir_elo = IsotonicRegression(out_of_bounds='clip').fit(oof_elo, y)
```
- **각 모델에 독립적으로** Isotonic Calibration 적용 (다른 노트북은 앙상블 후 1번만 보정)
- OOF 예측값 → 단조 변환 → 보정된 확률

### 10.6 앙상블 (자동 최적 가중치 탐색)

```python
for w in np.arange(0.0, 1.05, 0.05):
    bs = brier_score_loss(y, w × cal_lgb + (1 − w) × cal_elo)
```
- 0.05 간격으로 가중치 탐색 → Brier Score 최소화
- **최적 결과:** `LGB = 0.55`, `Elo = 0.45`, **Brier = 0.1682**

> LightGBM(0.1738) + Elo(0.1753) 각각보다 앙상블(0.1682)이 **5~7 포인트 개선**

### 10.7 예측 및 제출

#### 예측 파이프라인
1. **LightGBM 예측:** 5-fold 모델의 평균 → `clip(0.001, 0.999)` → Isotonic 보정
2. **Elo 예측:** `expit((Elo1 − Elo2) / 122.4)` → `clip(0.001, 0.999)` → Isotonic 보정
3. **앙상블:** `0.55 × cal_lgb + 0.45 × cal_elo`
4. **최종 클리핑:** `clip(0.05, 0.95)`

#### 성별 분리
- `Team1 ≥ 3000 or Team2 ≥ 3000` → 여성 (Gender=1), otherwise → 남성 (Gender=0)

#### NaN/Inf 처리
```python
lgb_raw = np.where(np.isfinite(lgb_raw), lgb_raw, 0.5)
elo_raw = np.where(np.isfinite(elo_raw), elo_raw, 0.5)
```
- NaN/Inf 값을 0.5 (중립)로 대체

#### 제출 통계
```
Rows: 519,144
Pred range: [0.050, 0.950]
Mean: 0.479
```

### 10.8 주요 설계 특징 및 비교

| 항목 | Notebook 10 (본 노트북) | 다른 노트북 |
|------|------------------------|-------------|
| **피처 수** | **13개** | 4~344개 |
| **팀 통계 집계** | **Recency 가중 평균** (RW) | 단순 평균 / 지수 감쇠 |
| **모델 구조** | **LightGBM + Elo 로지스틱** (2-model) | 단일 앙상블 또는 다중 트리 |
| **Elo 스케일** | **Scipy 최적화 (122.4)** | 고정 400 |
| **보정** | **각 모델별 독립 Isotonic** | 앙상블 후 단일 보정 |
| **앙상블 가중치** | **자동 탐색 (0.55/0.45)** | 수동, Grid Search, SLSQP 등 |
| **CV** | StratifiedKFold (무작위) | Temporal / StratifiedKFold |
| **남녀 처리** | **합산 + Gender 플래그** | 완전 분리 / 합산 |
| **여성 Massey 대체** | **NetEff rank** | 기본값 0 / NaN |
| **diff 부호 처리** | **LOWER_BETTER 집합으로 자동 반전** | 수동 / 일률적 T1−T2 |
| **OOF Brier (앙상블)** | **0.1682** | 0.05~0.18 |
| **클리핑** | [0.05, 0.95] | [0.001, 0.999] ~ [0.05, 0.95] |

### 10.9 종합 코멘트 (Review)

**장점:**
- **Recency 가중 평균**으로 시즌 후반 폼을 자연스럽게 반영 — 단순 `mean()`보다 정교
- **Elo 스케일 최적화** (`minimize_scalar`)로 확률 보정의 수학적 최적해 자동 탐색
- **LightGBM + Elo 이중 모델 앙상블**이 각 모델 단독보다 명확히 우수 (0.1738/0.1753 → 0.1682)
- **각 모델별 독립 Isotonic Calibration** — 앙상블 후 보정보다 세밀한 보정 가능
- **LOWER_BETTER 집합**으로 피처 부호 방향을 자동 관리 — 휴먼 에러 방지
- **True Shooting %**, **리바운드 점유율** 등 고급 통계 활용
- 여성 Massey를 `NetEff rank`로 대체하여 피처 결측 회피
- 단일 셀임에도 코드 구조가 13단계로 잘 분리됨

**한계점:**
- **StratifiedKFold (무작위 분할)** 사용으로 시간 기반 누수 가능 — Temporal CV가 더 적합
- 남녀 합산 학습이 성별 간 업셋 비율 차이(Men 27% vs Women 21%)를 충분히 반영하지 못할 수 있음 (Gender 피처가 부분적으로 보완)
- **SoS 산출에 NetEff를 사용하는데, NetEff 자체가 가중 평균** — 순환 참조는 아니나 동일 데이터 재사용
- Elo 산출에 **정규시즌만 사용** → 토너먼트 성과 미반영 (Notebook 1, 7은 토너먼트 포함)
- Early Stopping patience=50이 상대적으로 짧음 (Notebook 1: 75, Notebook 5: 100)
- Recency 가중치가 **시즌 내 선형**이라 시즌 초반 경기도 0.5 가중치로 무시되지 않음 — 지수 감쇠 대비 덜 공격적
- **OOF Brier 0.1682는 토너먼트 경기만 사용한 타 노트북(Notebook 1, 3의 ~0.05~0.11)과 직접 비교 불가** — 학습 데이터 구성(정규시즌 포함 여부)이 다름

---

## 11. march-ml-mania-2026-brierscore-modeling.ipynb
**주요 특징:** 전체 분석 노트북 중 **가장 극단적으로 단순한 파이프라인**. 단 **1개 피처(`WinPct_Diff`)**만으로 LightGBM 학습 → `CalibratedClassifierCV`(Isotonic, cv=5)로 보정. 학습 데이터가 **정규시즌 게임**(남녀 합산 28.6만행)이며, 시간 기반 분할(≤2021 train / >2021 val). Stage 2 제출용. Validation Brier **0.1639**.

### 11.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | 승률 산출 + **학습 데이터** |
| `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` | (읽지만 미사용) |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | (읽지만 미사용) |
| `MTeams.csv` / `WTeams.csv` | (읽지만 미사용) |
| `SampleSubmissionStage1.csv` | Stage 1 제출 양식 (519,144행) |
| `SampleSubmissionStage2.csv` | **Stage 2 제출 양식** (132,133행) |

> **핵심:** 토너먼트 데이터가 아닌 **정규시즌 전체 경기**를 학습 데이터로 사용하는 유일한 노트북. Seeds, Tourney, Teams를 읽지만 **피처에 활용하지 않음**.

### 11.2 전처리 및 피처 엔지니어링 (Feature Engineering)

> **총 1개 피처** — 모든 분석 노트북 중 최소

#### A. 남녀 합산
```python
regular = pd.concat([M_regular, W_regular])   # CompactResults
tourney = pd.concat([M_tourney, W_tourney])   # (미사용)
seeds   = pd.concat([M_seeds, W_seeds])       # (미사용)
```

#### B. 정준 순서 변환 (`prepare_games`)
```python
Team1 = np.minimum(WTeamID, LTeamID)   # 낮은 ID
Team2 = np.maximum(WTeamID, LTeamID)   # 높은 ID
Team1_win = (WTeamID == Team1).astype(int)
```
- **정규시즌** 전체 경기를 정준 순서로 변환

#### C. 승률 (`WinPct`) 산출
```python
Wins  = groupby(['Season','WTeamID']).size()
Losses = groupby(['Season','LTeamID']).size()
WinPct = Wins / (Wins + Losses)
```
- `CompactResults` 기반, 남녀 합산 정규시즌

#### D. 매치업 피처 (1개)

| 변수명 | 산출식 | 설명 |
|--------|--------|------|
| `WinPct_Diff` | `Team1_WinPct − Team2_WinPct` | 승률 차이 |

### 11.3 학습 데이터 구성

> **다른 모든 노트북과 근본적으로 다른 학습 데이터 전략**

#### 학습 데이터: 정규시즌 게임 (토너먼트 아님!)
```python
train = games[games['Season'] <= 2021]   # ~286,471행
valid = games[games['Season'] > 2021]    # ~57,295행 (2022~2026)
```
- **학습:** Season ≤ 2021의 정규시즌 경기 (286,471행)
- **검증:** Season > 2021의 정규시즌 경기

> **토너먼트 경기가 아닌 정규시즌 경기를 Target으로 사용** — 다른 노트북은 모두 `NCAATourneyCompactResults`를 학습 Target으로 사용

#### 양성 비율
- `pavg = 0.493498` → T1(낮은 ID)가 승리하는 비율 약 49.3%

### 11.4 모델 학습

#### LightGBM (단일 모델, CV 없음)

```python
model = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.01,
    max_depth=-1, num_leaves=64,
    subsample=0.8, colsample_bytree=0.8
)
model.fit(X_train, y_train)   # 검증셋 없이 전체 학습
```

| 파라미터 | 값 |
|----------|-----|
| `n_estimators` | 1000 |
| `learning_rate` | 0.01 |
| `max_depth` | -1 (무제한) |
| `num_leaves` | 64 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

> **Early Stopping 없음**, 검증셋 미전달. `num_leaves=64`는 피처 1개에 비해 매우 큼.

### 11.5 확률 보정 (`CalibratedClassifierCV`)

```python
calibrator = CalibratedClassifierCV(model, method="isotonic", cv=5)
calibrator.fit(X_train, y_train)
```
- **Sklearn 내장 CalibratedClassifierCV** 사용 — 직접 Isotonic 적용과 다른 방식
- 내부적으로 5-fold CV로 base model을 재학습 + 각 fold에서 Isotonic 보정기 학습
- **학습 데이터(X_train)에서 내부 CV** 수행 (229,176~229,177행/fold)
- 최종 예측: 5개 보정기의 **평균**

### 11.6 성능 결과

| 지표 | 값 |
|------|-----|
| Raw LightGBM Validation Brier | 0.16386 |
| Calibrated Validation Brier | 0.16388 |

> **보정이 오히려 미세하게 악화** (0.16386 → 0.16388) — 정규시즌 데이터로 학습한 단일 피처 모델에서 Isotonic이 효과 없음

### 11.7 예측 및 제출

#### 제출 과정 (Stage 2)
- 최초 Stage 2 제출 (132,133행) → 이후 Stage 1 제출 (519,144행)로 재작성
- 동일한 `calibrator.predict_proba(X_sub)[:,1]` 사용
- **클리핑 없음** — raw 보정 확률 그대로 제출

#### 제출 예시 (출력)
```
2026_1101_1102  0.820435
2026_1101_1103  0.060503
2026_1101_1104  0.164821
2026_1101_1105  0.393938
2026_1101_1106  0.618639
```
- 확률 범위: ~[0.06, 0.82] (클리핑 미적용)

### 11.8 주요 설계 특징 및 비교

| 항목 | Notebook 11 (본 노트북) | 다른 노트북 |
|------|------------------------|-------------|
| **피처 수** | **1개** (극소) | 4~344개 |
| **학습 데이터** | **정규시즌 경기** (286K행) | 토너먼트 경기 (1K~12K행) |
| **학습 대상** | 정규시즌 승패 | 토너먼트 승패 |
| **데이터 분할** | **시간 기반** (≤2021 / >2021) | CV / 시간 기반 |
| **모델** | LightGBM 단독 | 앙상블 (2~11개) |
| **보정** | **CalibratedClassifierCV** (sklearn 내장) | IsotonicRegression (직접) |
| **CV** | 보정기 내부 5-fold만 | 모델 자체 CV |
| **클리핑** | **없음** | [0.001, 0.999] ~ [0.05, 0.95] |
| **남녀 처리** | **합산, 구분 없음** | 합산+Gender / 완전 분리 |
| **Elo** | **미사용** | 거의 모든 노트북에서 사용 |
| **시드** | **미사용** | 거의 모든 노트북에서 사용 |
| **Massey** | **미사용** | 남성용 대부분 사용 |
| **제출 대상** | **Stage 2** (132,133행) → Stage 1 (519,144행) | Stage 1 |

### 11.9 종합 코멘트 (Review)

**장점:**
- **극도의 단순성** — 이해, 구현, 디버깅이 가장 용이
- **정규시즌 학습으로 대규모 데이터 확보** (286K행) — 토너먼트(~2K행)보다 100배 이상
- **Validation Brier 0.1639**는 다른 토너먼트 학습 노트북과 직접 비교는 어렵지만, 정규시즌 예측으로서 양호
- `CalibratedClassifierCV` 사용으로 코드가 간결
- Stage 1과 Stage 2 모두 제출 시도

**한계점:**
- **1개 피처는 정보 극히 부족** — Elo, 시드, 박스스코어, SoS 등 미사용
- **정규시즌으로 학습 → 토너먼트 예측**이라는 **도메인 불일치(Domain Shift)** — 정규시즌 승률이 토너먼트 승패를 직접 예측하기에는 분포가 다름
  - 정규시즌의 Team1_win rate ≈ 49.3% vs 토너먼트에서 낮은 ID의 승률 ≈ 50~52%
- `num_leaves=64`는 피처 1개에 과도 — 사실상 1D 히스토그램 분류기
- **토너먼트 데이터를 읽지만 미사용** — Seeds, Tourney 결과를 완전 무시
- Calibration이 효과 없음 (오히려 0.00002 악화) — 이미 잘 보정된 단순 모델
- **클리핑 없음** → 극단적 예측값(0.06 등)이 Brier 패널티 유발 가능
- 노트북 후반에 Stage 2 → Stage 1으로 **제출 형식을 시행착오**하는 과정이 그대로 남아 있어 코드 정리 부족

---

## 12. ncaa-2026-0-14722-tabpfn-time-series-cv.ipynb

**주요 특징:** 오프라인 환경에 맞춘 **TabPFN v2.5(Transformer 기반)** 파이프라인. 남성부는 `TabPFN`, 여성부는 `Logistic Regression`으로 완전히 분리하여 학습. **Time-Series CV(2021-2025)** 를 통해 검증하며, 자체적인 Ridge 기반 Team Quality 스코어와 Elo 등 복합적인 파생 피처를 사용. `IsotonicRegression` 보정 및 시드 기반 확률 블렌딩. Stage 1 제출용. Validation Brier는 남성 0.1847, 여성 0.1292.

### 12.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonDetailedResults.csv` ... | Box Score 통계(TS%, Ast, TO 등) 및 Team Quality 추출 |
| `MNCAATourneyCompactResults.csv` ... | 승패 결과 Target 및 학습/검증 기준 데이터 (Time-Series CV) |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 산출, 블렌딩 및 시드 차이 피처, 누적 토너먼트 경험 피처 |
| `MTeamConferences.csv` / `WTeamConferences.csv` | Power 6 컨퍼런스 소속 여부 확인 (`IsPower6` 추출) |
| `MMasseyOrdinals.csv` | Massey 랭킹 정보 산출 (Day 133 평균) - 남성부 한정 |
| `tabpfn-v2-5-full-weights...` (외부) | TabPFN v2.5 오프라인 가중치(Weights) 패키지 파일 |
| `SampleSubmissionStage1.csv` | 최종 Target 템플릿 (519,144행 예측) |

### 12.2 전처리 및 피처 엔지니어링 (Feature Engineering)

다양한 지표를 산출한 후 두 팀(T1, T2) 간의 **차이(diff)** 를 최종 특성으로 활용합니다.

#### A. 코어 지표 산출
```python
# 1. Team Quality (Ridge 기반 - Strength of Schedule 반영)
y = (season_df['WScore'] - season_df['LScore']).values
for i, row in enumerate(season_df.itertuples()):
    X[i, team_map[row.WTeamID]], X[i, team_map[row.LTeamID]] = 1, -1
model = Ridge(alpha=1.0).fit(X, y)

# 2. Elo Rating (과거 승리/패배 기록 누적 반영)
exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_elo) / 400.0))
elo[row.WTeamID] = w_elo + 20*(1-exp_w)

# 3. TS_pct (True Shooting Percentage - 2점, 3점, 자유투 효율 통합)
df['TS_pct'] = df['WScore'] / (2 * (df['WFGA'] + 0.44 * df['WFTA']).clip(1))
```

#### B. 성별 기반 매치업 피처 구성 (`FEATS_M` & `FEATS_W`)
학습/예측 시 T1 = min(ID), T2 = max(ID) 로 정렬한 후, `T1 - T2` 차이를 피처로 지정. 안정성 검증을 거쳐 성별로 다른 피처 세트를 사용합니다.

| 변수명 | 산출식 / 설명 | 사용 파이프라인 |
|--------|--------------|--------------|
| `Elo_diff` | Elo 레이팅 차이 | 공통 |
| `Qual_diff` | Ridge 기반 Quality Score 차이 | 공통 |
| `TS_diff` | True Shooting % 차이 | 공통 |
| `Seed_diff`| 시드 차이 | 공통 |
| `Ast_diff` | 어시스트(Ast) 평균 차이 | 공통 |
| `Rank_diff`| Massey 133일차 랭킹 차이 | **남성 전용 (`FEATS_M` - 총 6개)** |
| `TO_diff`  | 턴오버(Turnover) 평균 차이 | **여성 전용 (`FEATS_W` - 총 7개)** |
| `Exp_diff` | 누적 토너먼트 진출 횟수 차이 | **여성 전용** |

### 12.3 학습 데이터 구성 및 Time-Series CV 전략

```python
seasons = [2021, 2022, 2023, 2024, 2025]
for s in seasons:
    train, val = df[df.Season < s], df[df.Season == s]
```
- **특징:** 특정 연도(`s`)의 토너먼트를 검증(Validation) 셋으로 두고, 해당 연도 **이전까지의 모든 과거 토너먼트 데이터**를 Train 셋으로 사용하는 순차적 시계열 검증 방식 (Time-Series CV).
- 남성부와 여성부를 **완전히 분리**하여 별도의 데이터셋(`train_m`, `train_w`)으로 구성하고 독립적인 파이프라인을 실행합니다.

### 12.4 모델 학습 (Model Selection)

환경 감지 코드를 통해 오프라인 TabPFN 모델을 로드하며, 성별 특성에 맞춰 최적 모델을 선택합니다:
- **남성부 (Men):** `TabPFNClassifier` (Pre-trained Transformer를 이용한 In-Context Learning 적용). 스케일링 제외.
- **여성부 (Women):** `LogisticRegression(C=0.1, max_iter=1000)`. `StandardScaler` 적용.

> 코드 상단 설정(Configuration) 영역에서 `MEN_MODEL`과 `WOMEN_MODEL` 변수(CatBoost, XGBoost, LightGBM, RF 등)를 변경하는 것만으로 손쉽게 다른 앙상블 조합으로 교체할 수 있도록 확장성 있게 구현되어 있습니다.

### 12.5 확률 보정 및 앙상블 블렌딩 (Calibration & Blending)

예측 단계에서 여러 겹의 방어 수단을 적용하여 안정적인 확률을 생성합니다.

```python
# 1. Isotonic Regression으로 머신러닝 예측치(Out-Of-Fold) 확률 보정
calib = IsotonicRegression(out_of_bounds='clip').fit(oof_ml, labels)
cal_p = calib.transform(raw_p)

# 2. 통계적 역대 Seed 승률 (SEED_MAP) 병합
seed_p = sub_df.apply(lambda r: SEED_MAP.get(...)) 

# 3. 모델 보정확률(85%) + 시드 통계확률(15%) 블렌딩 및 클리핑
final_p = np.clip(0.85 * cal_p + 0.15 * seed_p, 0.025, 0.975)
```
- Sklearn 내장 함수 대신 OOF 예측들에 대해 명시적으로 `IsotonicRegression` 보정기를 피팅.
- 대회 특성을 고려한 **Seed Base 15% 앙상블** 적용.
- 예측 상/하한을 **[0.025, 0.975]** 로 제한하는 **하드 클리핑(Clipping)**으로 극단적 예측 실패 시 발생하는 막대한 Brier Score 패널티를 억제.

### 12.6 성능 결과 (Time-Series CV 2021-2025)

| 성별 | 사용 모델 | Time-Series Brier Score | 비고 |
|-----|-----------|-------------------------|------|
| **Men** | `TabPFN` | **0.1847** | ECE (Expected Calibration Error) 기록 자동 추출 |
| **Women**| `LogReg` | **0.1292** | - |

> 검증 후 `calibration_curve`를 통해 Raw 모델 확률 vs Calibrated 보정 확률을 선그래프(Reliability Diagram)와 대시보드로 시각화하여 신뢰도를 입증합니다.

### 12.7 예측 및 제출

- 남/녀를 구분하여 파이프라인을 두 번 실행:
  - `SampleSubmissionStage1` 데이터에 반복하여 `get_final_preds` 함수 적용.
  - Test 예측 과정에서도 전체 Train 데이터를 이용해 다시 Fit한 뒤 Test 데이터를 예측하고 Isotonic Transform → Seed Blending 15% → Clipping 공정을 동일하게 유지.
- `submission.csv` 총 519,144개 행 (Stage 1) 완성.

### 12.8 종합 코멘트 (Review)

**장점:**
- **TabPFN 딥러닝 도입** — 정형 데이터 대회에 Transformer 기반의 In-Context Learning이라는 참신하고 강력한 모델 구조를 시도.
- **견고한 검증 시계열 체계** — 단순 K-Fold가 아닌 Time-Series CV를 적용함으로써, 미래 시즌을 예측하는 본 대회의 Task에 가장 현실적으로 대응.
- **성별 맞춤 파이프라인** — 팀/리그간 통계 상관관계가 다름(남성은 랭킹, 여성은 경험과 턴오버가 더 유의미함)을 반영하여 피처 리스트(`FEATS_M`/`FEATS_W`)와 모델 자체를 분리한 세밀함.
- **철저한 보완 장치** — `Isotonic` 캘리브레이션, 15% `Seed` 블렌딩 앙상블, 하드 클리핑을 삼중으로 적용하여 업셋 등 이상 변동에 리스크를 방어.

**한계점:**
- TabPFN의 학습 시 구조적 연산량 한계 — 대형 데이터셋에 사용하기 어려울 수 있으며 로컬 오프라인 우회 환경 설정이 복잡함.
- Stage 2(13만행) 예측 형태로는 바로 적용되어있지 않아 템플릿 로드 수정 필요 가능성 존재.

---

# [2025년 대회 예시 데이터 분석]

## 1. a-more-aggressive-leap.ipynb (2025)

### 개요
이 노트북은 2025년 대회 데이터 구조를 기반으로 **RandomForestRegressor** 단일 모델 체제를 구축한 Baseline 코드입니다. 본 대회(2026년) 역시 2025년과 동일한 포맷을 공유할 가능성이 높으므로 가장 기본적인 파이프라인의 구조를 파악하기 좋습니다. 가장 큰 특징은 남/녀 토너먼트 데이터를 분리하지 않고 완전히 하나로 병합하여 학습하며, 예측의 안정성을 위해 분류(Classification)가 아닌 회귀(Regression) 방식을 적용한 뒤 최종적으로 `IsotonicRegression` 보정을 거친다는 점입니다.

### 1.1 데이터 로드 및 전역 병합 (Data Loading & Concatenation)
이 커널은 클래스 `TournamentPredictor` 하나로 전체 파이프라인을 감쌉니다. 파일 병합의 특징은 `M(Men)`과 `W(Women)` 접두사로 분리된 데이터를 모두 읽어 그대로 행(Row) 방향으로 붙여버린다는 것입니다.

```python
# Process teams and team spellings (남/녀 통합)
teams = pd.concat([self.data['MTeams'], self.data['WTeams']])
teams_spelling = pd.concat([self.data['MTeamSpellings'], self.data['WTeamSpellings']])

# Concatenate season and tournament results
season_cresults = pd.concat([self.data['MRegularSeasonCompactResults'], self.data['WRegularSeasonCompactResults']])
season_dresults = pd.concat([self.data['MRegularSeasonDetailedResults'], self.data['WRegularSeasonDetailedResults']])
tourney_cresults = pd.concat([self.data['MNCAATourneyCompactResults'], self.data['WNCAATourneyCompactResults']])
tourney_dresults = pd.concat([self.data['MNCAATourneyDetailedResults'], self.data['WNCAATourneyDetailedResults']])
```
이러한 접근법은 모델이 학습할 수 있는 **데이터 포인트의 양을 두 배로 늘리는 효과**가 있습니다. 특히 트리 모델계열(Random Forest 등)은 데이터 모수가 많을수록 분산이 줄어들고 더 복잡한 패턴을 분리해 낼 수 있기 때문에, 성별 차이 규칙보다는 농구라는 스포츠 고유의 통계적 규칙(예: FG%, 리바운드 우위가 승리에 미치는 영향)을 하나로 통합하여 가르치려는 의도입니다.

### 1.2 핵심 피처 엔지니어링: 매치업 집계 (Matchup Aggregation)
데이터 준비 과정에서 이 노트북의 가장 독특하고 강력한(동시에 다소 위험한) 피처 엔지니어링이 등장합니다.

**첫째, 팀 정렬 및 IDTeams 키 생성:**
경기 승리/패배(`WTeamID`, `LTeamID`) 컬럼 대신, 언제나 ID 번호가 작은 쪽을 `Team1`, 큰 쪽을 `Team2`로 배치합니다. 그리고 이를 언더바(`_`)로 이은 `IDTeams` 변수를 만듭니다.

```python
self.games['IDTeams'] = self.games.apply(
    lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1
)
```

**둘째, 역대 모든 Detailed Box Score 통계 병합:**
정규시즌과 토너먼트의 **모든 Box Score 데이터**(FGM, Ast, TO 등 총 27개 지표)를 `IDTeams`(두 팀의 역대 매치업) 단위로 묶어 다수의 통계치를 생성합니다.

```python
c_score_col = [
    'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 
    'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 
    'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
]
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']

# 역대 매치업(IDTeams)별로 앞선 27개 변수에 대해 8개의 통계함수를 전부 적용
self.gb = self.games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
self.gb.columns = [''.join(c) + '_c_score' for c in self.gb.columns]
```
이렇게 만들어진 통계 테이블을 학습 셋(토너먼트 경기 한정)과 테스트 셋(Submission)에 Left Join 합니다. 즉, 모델은 예측해야 할 두 팀이 지난 20년간 맞붙었을 때 보여줬던 평균 득점력, 최대 점수차, 리바운드 분산 등을 한 번에 피처로 입력받습니다. 

> *주의: 이 방식은 2018년 경기를 예측할 때 2024년에 둘이 만난 경기 결과까지 누적된 평균이 스며들게 되는 Data Leakage (미래 정보 참조 오류)가 발생합니다.*

### 1.3 머신러닝 아키텍처 (Random Forest + Isotonic Regression)

이 노트북은 이진 분류임에도 변동성을 줄이기 위해 回歸(Regressor) 알고리즘을 사용합니다. 결측치는 평균값으로 채우고(`SimpleImputer`), `StandardScaler`를 통과합니다.

**RandomForestRegressor 세팅:**
```python
self.model = RandomForestRegressor(
    n_estimators=250, 
    random_state=42, 
    max_depth=20,          # Depth 15 이상 시 변동성 증가 우려. 20으로 타협
    min_samples_split=3,   # 스플릿 최소 샘플수를 올려 과적합 제어
    max_features='sqrt'    # 트리의 다양성 확보
)
```

학습 완료 후 모델 출력값(pred)은 0.0 ~ 1.0 사이의 연속형 값이 됩니다. 이후 Brier Score 최적화를 위해 **IsotonicRegression**을 후처리기로 추가 결합합니다.

```python
# Train-Set 예측치를 활용해 Calibration 보정 모델 적합
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(pred, y)

# Prediction 적용 시: RF 모델의 Test Set 예측치를 Isotonic으로 한 번 더 펼침
preds = self.model.predict(sub_X_scaled).clip(0.01, 0.99)
preds_cal = ir.transform(preds)
```

이중 보정을 거치는 이유는 RF 분류기로 뽑은 `predict_proba`가 중간값(0.5) 근처로 과하게 모이는 현상(Distribution collapse)을 방어하고 극단값에 대해 신뢰도 구간을 매핑해주기 위함입니다. 최종 제출단에서도 값을 한 번 더 `[0.01, 0.99]`로 자릅니다(하드 클리핑).

### 1.4 주요 성능 평가 (Training & Cross-Validation)

| 평가지표 | 모델 자체 성능 (Train-fit) | Cross Validation 성능 (5-Fold) |
|--------|----------------------------|--------------------------------|
| **Log Loss** | 0.0072 | - |
| **Brier Score** | 0.0021 | - |
| **MSE**| - | 0.1944 |

데이터 Leakage로 인한 통계 피처 때문에 초기 Fit된 모델은 Train 셋을 거의 배기량 100% 흡수하여 과적합 수준의 Brier (0.002)를 달성합니다. 반면 순수 K-Fold CV를 돌려보면 평균 에러(MSE)는 0.19 수준으로 회귀하며 이는 준수한 Baseline 수준의 로버스트함(Robustness)을 보여줍니다. 결과물인 `submission.csv`는 총 **519,144개 행**에 대해 생성됩니다.

### 1.5 인사이트 및 종합 평가 (Review)

**장점 (Strengths):**
1. **데이터 단일화의 편의성:** M과 W를 구분하지 않고 Concat을 사용하여 전처리 코드를 절반으로 줄이고, 단일 피처 임퓨팅과 스케일링을 관리하기 편하게 만든 우수한 템플릿입니다.
2. **풍부한 스탯 압축 기법:** 27개의 변수를 8가지 통계 함수(`mean, sum` 등)와 결합하여 무려 216개 이상의 피처 공간을 팽창시켜 트리 모델의 판단 근거를 넘치게 제공했습니다.
3. **분류(Classification) 대신 회귀(Regression) 역발상 사용:** 과적합이 쉬운 Classification 트리 대신 Regressor를 통해 확률 스코어를 뽑고 Isotonic으로 보완하는 방식은 Brier Score 평가지표에 대한 이해도가 높은 테크닉입니다.

**한계 및 주의점 (Limitations):**
1. **Data Leakage (미래 누수 참조):** 앞서 언급했듯 전체 데이터를 통째로 `.groupby` 하여 평균을 내면 과거를 예측할 때 미래 이벤트가 반영되므로, 과거 연도를 검증(Validation) 셋으로 분리하는 고도화된 Time-Series 방식을 해당 코드만으로는 수행하기 무척 까다롭습니다.
2. **극단값 하드 클리핑(`clip(0.01, 0.99)`):** 업셋(이변)을 고려하지 못하고 모델이 완전히 틀렸을 때 발생하는 Brier 공식 페널티 폭탄을 피하기 위해 임의로 확률을 잘라내는 방식입니다. 확률론적으로 약간은 보수적인 편이라 리더보드 상위권 도약을 위한 엣지(Edge)가 조금 부족해질 수 있습니다.

---

## 2. baseline.ipynb (2025)

**주요 특징:** 2025년 대회를 겨냥한 **"Improved Pipeline"**으로, 다중 모델 앙상블과 메타 러닝, 확률 보정까지 모든 기법을 동원한 모범적인 베이스라인입니다. 승패 이진 분류 외에 점수차를 연속형으로 예측하는 **Margin Regression(Ridge)** 을 앙상블 피처로 추가 활용한 점이 돋보입니다. 또한 무지성 평균이 아닌, 경기 페이스(Tempo)를 고려한 **포제션 기반 고급 스탯(Advanced Stats)** 과 **동적 Elo Rating**, 그리고 **다중공선성(VIF) 검증**을 통해 양질의 28개 피처만으로 압축했습니다.

### 2.1 사용 데이터

| 파일명 | 용도 |
|--------|------|
| `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` | 동적 Elo Rating 누적 계산, 최근 14일 모멘텀 산출 + **학습 데이터 구성** |
| `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` | 포제션(Possession) 기반 2차 농구 스탯 산출 |
| `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` | **최종 Target(학습 데이터)** |
| `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` | 시드 번호 산출 |
| `MTeamConferences.csv` / `WTeamConferences.csv` | Conference Strength (소속 컨퍼런스 팀들의 평균 Elo) 산출 |
| `MMasseyOrdinals.csv` | Massey 랭킹 133일차 시스템 평균 (남성부 한정) |
| `SampleSubmissionStage2.csv` | Stage 2 제출용 예측 그리드 |

> **핵심:** 정규시즌 데이터로 각종 스탯과 Elo를 깎고, 학습의 **Target 셋으로는 토너먼트 데이터(M, W_tourney_compact)** 와 더불어서 일부 정규시즌 후반 경기(`frac=0.15`)를 표본 추출해 함께 학습시킨 특징이 있습니다.

### 2.2 전처리 및 피처 엔지니어링 (Feature Engineering)

> **총 28개 피처** — 농구 도메인 지식의 총집합

#### A. 동적 Elo Rating 계산 (`compute_elo_ratings`)
- 시즌별로 승패뿐만 아니라 **점수차(Margin of Victory)** 와 **홈 코트 어드밴티지(`ELO_HOME_ADV = 100`)** 를 고려하여 Elo 포인트를 등락시킵니다.
- 새 시즌이 시작될 때는 `ELO_CARRYOVER = 0.75` 비율로 이전 시즌의 실력을 계승합니다.

#### B. 포제션 기반 2차 스탯 정규화 (`compute_advanced_stats`)
과거 총 득점이나 실점 등은 다중공선성(VIF) 문제가 너무 커서 모델 학습을 방해하므로, Oliver의 포제션 공식을 입혀 100회 공격권 당 득점(ORtg) 등으로 변환합니다.

```python
# 1. 공격권(Possession) 추정 공식
combined["poss"] = combined["fga"] + 0.475*combined["fta"] - combined["off_reb"] + combined["to_"]

# 2. 고급 스탯 변환 (일부)
agg["off_rtg"]       = 100 * agg["total_pts"]     / p
agg["def_rtg"]       = 100 * agg["total_opp_pts"] / p
agg["net_rtg"]       = agg["off_rtg"] - agg["def_rtg"] # 핵심: 효율 차이
agg["efg_pct"]       = (agg["total_fgm"] + 0.5*agg["total_fgm3"]) / agg["total_fga"]
agg["true_shoot_pct"]= agg["total_pts"] / (2*(agg["total_fga"]+0.475*agg["total_fta"]))
agg["to_rate"]       = agg["total_to"]   / p
```

#### C. 매치업 피처 구성 (T1 - T2 차이값)
단일 팀 스탯이 아니라, 무조건 **(Team1 스탯 - Team2 스탯)** 형식의 `_diff` 변수로 뺄셈하여 상대적 우위만 모델에 전달합니다 (총 28개 차이값 변수). 여성부의 경우 Massey 데이터가 없으므로 `avg_massey_rank`를 최하위 랭킹(300)으로 결측치 처리합니다.

### 2.3 학습 데이터 구성

> **정규시즌 데이터 일부 편입 + 타겟 레이블 설정**

#### 학습 대상 분리
```python
def add_game(season, wa, lb, a_won):
    t1, t2 = min(wa,lb), max(wa,lb)
    # t1과 t2의 _diff 피처 추가
    feat["target"] = 1.0 if (a_won and wa==t1) or (not a_won and lb==t1) else 0.0
```
- 모든 NCAATourney 경기를 학습에 추가합니다.
- 토너먼트 수 부족을 상쇄하기 위해 정규시즌 후반 게임(`DayNum >= 100`) 중 무작위 15%(`frac=0.15`) 샘플을 토너먼트 특성과 비슷하다고 보고 추가합니다.
- **최종 Rows:** 12,242행 (Target 분포: 0.496)

### 2.4 모델 학습 및 앙상블 (2단계 Stacking)

단일 모델이 아닌 XGBoost, LightGBM, CatBoost를 기반 모델로 하고 메타 모델을 씌웁니다. 사전에 Optuna를 통해 `OPTUNA_TRIALS=60` 만큼 파라미터를 튜닝하여 얻은 `best_params` 를 활용합니다.

#### 1단계: Base Models (5-Fold CV, Random Seeds = [42, 123, 2026])
동일한 KFold 내에서 세 모델을 동시에 돌리며 예측 확률값을 저장합니다.

| 모델 | 최적 파라미터(예시/기본) | 목적 / 설명 |
|------|-----------|-------------|
| **XGBClassifier** | `n_estimators=600`, `max_depth=4`, `learning_rate=0.02`, `tree_method="hist"` | 보편적 부스팅 베이스라인 |
| **LGBMClassifier** | `n_estimators=1000`, `num_leaves=16`, `learning_rate=0.02`, `min_child_samples=30` | `num_leaves`를 크게 주면 과적합되므로 16으로 강한 제재 |
| **CatBoostClassifier** | `iterations=800`, `depth=5`, `learning_rate=0.02` | 기본 성능이 우수한 Categorical Booster |

#### 2단계: Margin Regression (Ridge 회귀 병합)
단순 이진분류(승패)로는 알 수 없는 "얼마나 이겼는가" 정보를 살리기 위해, `target`이 아닌 **점수차(`margin`)를 Y값으로 두는 Ridge 회귀 모델(`alpha=10.0`)**을 병렬로 훈련합니다. 구해진 점수차 예측값은 시그모이드(`margin_to_prob`)를 거쳐 가상의 '확률'로 변환됩니다.

#### 3단계: Stacking Meta-Learner (Logistic Regression)
앞선 1단계의 부스팅 확률 3개(`oof_xgb, oof_lgb, oof_cat`)와 마진 변환 확률(`ridge_oof_prob`) 총 4개의 예측값을 새로운 입력(X)으로 삼아 `LogisticRegression(C=0.5)` 메타 모델을 학습시킵니다.
* 모델 간 중요도 가중치 결과: XGB: 1.098 / LGB: 1.065 / CAT: 1.646 / **Ridge: 2.915** (점수차 모델이 의외로 가장 중요하게 쓰임을 알 수 있습니다.)

### 2.5 확률 보정 (`IsotonicRegression`)

메타 모델을 통과해서 나온 최종 결과물(`final_oof`)을 Scikit-learn의 `IsotonicRegression`에 넣어 S-Curve 왜곡 현상을 매끈한 사선으로 교정합니다.

```python
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(final_oof, y_true)
cal_oof = iso.predict(final_oof)
```

### 2.6 성능 결과

Kaggle 로그 상의 출력 성능입니다. Leave-One-Season-Out (LOSO) 5년 단위 검증도 수행합니다.

| 지표 | 로그 손실 (LogLoss) |
|------|--------------------|
| 평균 LOSO (2021~2025) CV | 0.47417 |
| 메타 모델(Stacking OOF) | 0.45764 |
| **Isotonic 캘리브레이션 이후 OOF** | **0.44941** |

> 모델 앙상블로 0.45대까지 낮추고, 캘리브레이션을 통해 0.449까지 영혼까지 끌어올린 지표입니다.

### 2.7 예측 및 제출

- 남녀 통합 Stage 2 제출 (131,407행)
- `cal_preds = iso.predict(final_probs)` 후 `np.clip(cal_preds, 0.05, 0.95)`를 사용해 극단적인 Brier 점수 패널티를 하드 클리핑으로 마이너하게 제어하여 제출합니다.

### 2.8 주요 설계 특징 및 비교

| 항목 | Notebook 2 (본 노트북) | 다른 단순 모델 |
|------|------------------------|-------------|
| **피처 수** | **28개** (품질 관리됨) | 매우 적거나 지나치게 많음 |
| **학습 데이터** | **토너먼트 + 일부분 정규시즌** | 정규시즌 전체 혹은 토너먼트만 |
| **학습 대상** | 승패 이진 분류 + **마진 회귀 추가** | 승패 이진 분류만 |
| **모델** | **XGB + LGB + Cat + Ridge 스태킹** | LightGBM 등 단일 모델 |
| **보정** | **IsotonicRegression** | 안 하거나 내장 함수 이용 |
| **남녀 처리** | **학습 직전에 병합** | 병합 혹은 분리 |
| **Massey** | **사용 (결측치는 300 우회)** | 미사용 |

### 2.9 종합 코멘트 (Review)

**장점:**
- **농구 도메인 지식의 정수** — 단순 평균 득점이 아닌 eFG%, ORtg, Pace 등 농구 전문 스탯을 직접 계산하였으며 VIF 다중공선성을 철저하게 회피했습니다.
- **점수차 기반 로지스틱 변환 앙상블** — 회귀 모델의 예측값을 로지스틱 인풋으로 넣어 메타 러닝의 핵심 뼈대로 삼은 테크닉(순수 트리보다 Ridge 스코어 가중치가 더 높음)은 매우 강력합니다.
- **다중 부스팅 앙상블과 캘리브레이션의 정석** — 5-Fold, 3개 시드 기반 Randomness 제어, Isotonic 보정까지 완벽한 방어를 수행했습니다.

**한계 및 주의점 (Limitations):**
- **데이터 Leakage 여부 방어 필요:** `compute_advanced_stats` 에서 `Season` 단위로 `.groupby`를 묵어서 평균을 도출하므로, 초반 게임을 예측할 때 나중 경기의 성패가 반영되는 약소한 Leakage가 있습니다.
- **남녀 데이터 합산 시의 이질감 방치** — 여성부에는 없는 Massey Ordinals 랭킹 데이터를 강제로 300위(최하위권)로 결측채움하여 남녀 데이터를 하나의 모델에 욱여넣다 보니, 남녀 경기 양상 차이가 교란을 줄 위험이 내재되어 있습니다.

---

## 3. isotonic-calibaration-with-all-features.ipynb (2025)

**주요 특징:** 대회에서 상위권에 랭크된 파이프라인으로, 앞선 `baseline.ipynb`의 아이디어를 극대화하여 활용도가 높은 파생 피처(Derived Features)들을 대거 포함시킨 완성형 코드입니다. 특히 `Massey Ordinals`(KenPom 등 랭킹) 데이터, 시즌 후반(최근 14일)의 모멘텀 지표, 그리고 전체 정규시즌 승률과 마진을 피처 세트에 녹여냅니다. 학습 단계에서는 **XGBoost, LightGBM, CatBoost 뿐만 아니라 ExtraTreesClassifier를 포함한 4종 블렌딩(Blending)**으로 앙상블 시스템을 구축하며, 무수히 늘어난 독립 변수들 사이의 충돌을 방지하기 위해 **VIF(Variance Inflation Factor)와 OLS p-value 기반의 통계적 피처 선택(Feature Selection)** 과정을 도입한 점이 가장 수훈갑으로 꼽힙니다. 검증과 타겟 보정 단계 역시 Isotonic Regression을 통해 세밀하게 통제하고 있습니다.

### 3.1 사용 데이터 병합 방식

이 노트북은 거의 모든 가용한 대회 제공 CSV 파일을 불러온 뒤, 남녀부(M/W) 구분을 없애고 위아래로 병합(`pd.concat`)하여 단일 데이터프레임으로 처리하는 유연한 투트랙 전처리를 수행합니다.

| 파일명 (M/W 공통 결합) | 파생 피처 스키마 및 용도 |
|----------------------|---------------------------------------|
| `*RegularSeasonCompactResults.csv` | 동적 Elo Rating 누적, 최근 14일 모멘텀, 시즌 전체 승/패 비율 및 점수 마진 산출 |
| `*RegularSeasonDetailedResults.csv` | 2차 스탯(Advance Stats) 평균치 연산, 역대 맞대결(H2H)의 누적 집계 스탯 베이스라인 |
| `*NCAATourneyCompact`/`Detailed` | 토너먼트 본선 기록으로 **학습 타겟 데이터 분리 및 검증 피벗** 용도 |
| `*NCAATourneySeeds.csv` | 두 팀의 Seed 차이(`SeedDiff`)와 비율(`seed_ratio`) 계산용 |
| `*TeamConferences.csv` | `ConfAbbrev` 기준으로 소속 컨퍼런스의 평균 힘(`conf_elo`)을 구하기 위해 조인 |
| `MMasseyOrdinals.csv` | `KPK`(KenPom) 시스템의 랭킹 데이터 수집 (여성부 부재로 결측치 처리 필요) |
| `SampleSubmissionStage2.csv` | Stage 2 평가에 필요한 모든 가능한 매치업 (132,133행) 제출 템플릿 로딩 |

> **데이터 전략 핵심:** 남녀 모델을 분리하지 않고 합산(+`concat`) 학습하여 트리가 성별 간의 공통적인 농구 승리 메커니즘을 함께 배울 수 있도록 유도합니다. 

### 3.2 정교한 전처리 및 다중 피처 엔지니어링 (Feature Engineering)

모델 성능 향상의 8할을 차지하는 광범위한 피처 엔지니어링 과정을 거칩니다. 크게 팀 수준 피처, 매치업(Diff) 피처, 맞대결(H2H) 피처로 뼈대가 나뉩니다.

#### A. 동적 지표 생성 (Dynamic Elo & Conference Strength)
- **Elo Ratings:** 경기별로 시간 순서대로 훑으며(`sort_values(["Season", "DayNum"])`) 승패 마진과 홈/어웨이 어드밴티지를 반영해 Elo 점수를 갱신합니다. 매 시즌 시작 시 이전 시즌 Elo의 75%를 유지하는 오토 캐리오버(Carry-over) 로직을 적용해 연속성을 가져갑니다.
- **Conference Elo:** 개별 팀의 Elo 점수를 해당 연도의 컨퍼런스 단위로 묶어 평균화한 `conf_elo` 피처를 만듭니다. 약팀이더라도 강한 컨퍼런스 소속일 경우 보정을 기대할 수 있습니다.

#### B. 효율성 2차 스탯 (Advanced Efficiency Stats) 및 모멘텀 (Momentum)
- **Box Score 파생 지표:** 단순히 점수가 아닌 포제션(Poss) 기반의 고도화된 스탯을 뽑아냅니다.
  - `Poss = FGA + 0.475*FTA - OR + TO`
  - `eFG` (유효슈팅률), `TOR` (턴오버율), `OffRating` (오펜시브 레이팅, 포제션당 득점력), `FT_rate` (자유투 획득 비율), `3PA_rate` (3점슛 의존도)
- **Ranking (KenPom):** 남성부 `MMasseyOrdinals`에서 `KPK` 시스템의 가장 늦은 날짜 랭킹을 가져와 `KPK_rank`로 뺍니다. 여성부 데이터에는 해당 시스템이 없어 전처리 하단에서 일괄적으로 100(위)으로 강제 결측(Fillna) 채움을 진행합니다.
- **Late-Season Momentum:** 시즌 막판의 기세(DayNum >= 118)만을 필터링해 `win_ratio_14d`를 산출합니다.
- **Season Overall:** 정규 시즌 전체의 총 승률(`win_pct`)과 평균 마진(`w_margin`)을 산출합니다.

#### C. 포괄적인 Head-to-Head (H2H) 매커니즘
`a-more-aggressive-leap.ipynb` 처럼 역대 두 팀이 맞붙었던 모든 경기의 Box Score(`IDTeams`)를 한데 모아 폭력적인 Aggregation을 수행합니다.
```python
c_score_col = ['NumOT','WFGM','WFGA', ...] # 27개 Box 스탯
c_score_agg = ['sum','mean','median','max','min','std','skew','nunique'] # 8개 통계분포 함수
gb = all_games.groupby("IDTeams").agg({k: c_score_agg for k in c_score_col})
```
이 과정을 통해 순식간에 `27 * 8 = 216`개의 H2H 전용 피처가 탄생합니다.

#### D. **다중공선성(VIF) 억제 및 통계적 피처 선택 (Feature Filtration)**
엄청나게 불어난 피처(`Team1지표 - Team2지표`로 생성된 Diff 피처) 데이터베이스를 모델에 통짜로 넣으면 다중공선성과 노이즈로 트리가 붕괴됩니다. 이를 막기 위해 다음 필터링을 거칩니다:
1. `max_vif > 15.0` 인 피처들을 순회하며 가장 높은 변수부터 반복적으로 삭제합니다.
2. 살아남은 Diff 변수들을 모아 타겟(`target`)에 대한 OLS 선형 회귀 모형을 적합시킵니다.
3. 거기서 나온 독립 변수 중 **p-value가 0.05 미만으로 통계적 유의성이 검증된 피처만**을 남깁니다.
4. 이렇게 통과한 유효 Diff 피처와 앞선 H2H 피처를 합산하여 **최종 학습 피처 군을 222개로 한정**합니다. 매우 과학적인 접근법입니다.

### 3.3 학습 데이터 구성 (Oversampling & Concatenation)

#### 유연한 학습 대상 분리
흥미롭게도 토너먼트 본선 결과만 가지고 학습하면 데이터 수가 부족하므로, **정규시즌 후반 게임(`DayNum >= 100`) 결과 중 무작위 15%를 떼어내 토너먼트 데이터에 붙여(Concat)** 학습 데이터 볼륨을 의도적으로 확장시킵니다.
```python
train_raw = pd.concat([
    tourney_comp,
    reg_comp[reg_comp['DayNum'] >= 100].sample(frac=0.15, random_state=42)
], ignore_index=True)
train_data = train_data[train_data['Season'] >= 2010] # 2010년 이후만 사용
```

### 3.4 4대 모델 앙상블 시스템 및 캘리브레이션

#### 1단계: 모델 블렌딩 (3개년 검증)
하이퍼라미터 최적화 없이 지정된 구성으로, 가장 최근 3개 년도(`2024년`, `2023년`, `2022년`)를 바꿔가며 Validation 룹을 순회합니다. 총 4개의 트리/부스팅 모형을 가동합니다.

| 분류 모델 (Classifier) | 하이퍼파라미터 셋업 | 앙상블 기여 목적 |
|----------------------|-------------------|----------------|
| **XGBoost** | `n_estimators=600`, `max_depth=4`, `learning_rate=0.01` | 견고한 베이스라인 확률 예측 및 주요 피처 스플릿 최적화 |
| **LightGBM** | `n_estimators=600`, `max_depth=4`, `learning_rate=0.01` | 빠른 리프 확장 및 부스팅 성능 방어 |
| **CatBoost** | `iterations=600`, `depth=4`, `learning_rate=0.01` | 범주형 데이터 변이 및 미세 노이즈 제어 보강 |
| **ExtraTrees** | `n_estimators=250`, `max_depth=19`, `min_samples_split=3` | 무작위성을 극한으로 줘서 개별 부스팅 트리들의 Overfitting 상쇄 (상대결정트리) |

검증 시 4개 모델에서 나온 확률 변수를 단순 일대일 비율로 결합해 평균 확률(`(xgb+lgb+cat+et)/4`)을 냅니다.
*2024년 기준 Brier Score 도출 예:*
- XGB (0.1503), LGB (0.1503), CAT (0.1504), ET (0.1857) → **Blended = 0.1526**

#### 2단계: Isotonic 기법을 활용한 후처리 분포 캘리브레이션
앙상블 확률값의 우하향, 좌상향 꼬리(Probability Distortion)를 펴주기 위한 Isotonic Regression 교정이 이 노트북의 꽃입니다.
```python
ir_cal = IsotonicRegression(out_of_bounds="clip")
ir_cal.fit(ensemble_train, y_train_all) 
cal_train = ir_cal.transform(ensemble_train)
```
하지만 코드 상 **Fold 바깥(Out-of-Fold)에서의 검증된 확률값을 쓰는 것이 아니라 전체 합산 Training Set 예측값을 넣어 그대로 피팅**시킵니다. 이는 학습셋에 과도하게 최적화된 편향된 곡선을 도출할 여지를 남깁니다.
최종 제출물 생성 시에도 도출된 곡선에 예측 확률을 태운 뒤, 극단값 차단 처리인 `np.clip(..., 0.05, 0.95)`를 씌워 안정화 후 제출 포맷으로 내보냅니다.

### 3.5 종합 평가 리뷰 (Review)

**장점과 시사점 (Strengths):**
- **가장 진보된 피처 최적화 도구 사용:** 머신러닝만 돌리는 것이 아니라 선형 회귀의 전통적인 검증 방식인 **VIF 통제 기법과 다중공선성 p-value 하차 로직**을 활용하였습니다. 무분별하게 차원이 늘어난 H2H + Box 스탯 데이터 사이에서 핵심만 골라내는 매우 훌륭한 파이프라인 아키텍처입니다.
- **최신 도메인 트렌드 반영 (Momentum / Massey):** 시드와 Elo에만 국한되지 않고, 시즌 후반(14일)의 반등률 지표와 켄폼(Massey) 위상을 모델에 녹이면서 정규시즌 성적 대비 폼이 좋은 '업셋(Upset)' 잠재력 팀을 예측할 수 있는 기반을 다졌습니다.

**숨겨진 리스크 및 한계점 (Limitations):**
- **Extra Tree의 팀킬 여부:** Brier 점수 로그를 확인하면 XGB/LGB/CAT 모델은 안정적인 0.15 극초반대 점수를 수확하고 있으나, ExtraTrees 계열은 혼자 0.18을 훌쩍 넘기며 오차를 폭발시키고 있습니다. 블렌딩에 악영향을 주고 있을 거란 매우 강한 심증이 듭니다.
- **Isotonic Calibration의 Target Overfitting:** Test Set을 위한 보정 곡선을 만들 때 분리된 홀드아웃 셋이나 OOF 방식의 블라인드 확률이 아닌, 자신이 학습한 Train Set의 `target`을 이용해 Isotonic 커브를 Fit(`ir_cal.fit`) 해버렸습니다. 이는 커브 곡선이 학습 셋(Train)의 패턴에 완벽하게 오버피팅되어 실전 테스트 대응력이 반감될 수 있습니다.

---

## 4. march-machine-learning-mania-2025-olaf-laitinen.ipynb (2025)

**주요 특징:** 대회에서 클래식하게 쓰이던 정통 트리 모델 파이프라인의 전형을 보여주는 훌륭한 교보재 노트북입니다. 복잡한 외부 지표(Elo, Massey, 모멘텀 등)를 모두 배제하고, 오프라인 경기 박스스코어 27개 지표에만 기대어 역대 **해드투해드(H2H, 두 팀 간 누적 맞대결 통계량)**만을 파고들어 피처화한 것이 핵심입니다. 모든 프로세스를 `TournamentPredictor`라는 하나의 파이썬 클래스 객체 내에 객체 지향적(Object-Oriented)으로 결합하여 가독성, 유지보수성, 실행 속도를 극대화했습니다. 최적화 알고리즘의 유행이 부스팅 모델로 넘어온 시점에도 꿋꿋이 정통 **RandomForestRegressor**의 깊이 제어 규제를 사용하여 Brier 스코어를 안정적으로 억눌러낸 아키텍처입니다.

### 4.1 기본 사용 데이터 및 클래스 구조
| 파일명 | 로딩 형태 (M/W 통합) | 산출 용도 |
|--------|---------------------|---------|
| `*Teams` 및 `TeamSpellings` | `pd.concat` 후 팀 이름 스펠링 종류 수 집계 | `TeamNameCount` (팀 이름 변경 횟수) 파생 |
| `*RegularSeasonDetailed` | `pd.concat`으로 통합 조회 | 정규 시즌 H2H Box 통계량 |
| `*NCAATourneyDetailed` | 상동 | 토너먼트 본선 H2H Box 통계량 및 Target 데이터 |
| `*NCAATourneySeeds` | 매치업 팀 시드 배정 딕셔너리(`seeds_df`) | `SeedDiff` (두 팀 간 시드 차이) 산출 |
| `SampleSubmissionStage1.csv` | 타겟 프레임 로딩 | Stage 1 제출 양식용 ID 파싱 |

```python
class TournamentPredictor:
    def __init__(self, data_path): ...
    def load_data(self): ...
    def train_model(self): ...
    def predict_submission(self, output_file='submission.csv'): ...
    def run_all(self): ...
```
> **특징:** 도메인 전문가적인 어프로치(복잡한 Elo 계산 등)를 완전히 삭제하고, 오직 식별자, 시드, 그리고 과거 스탯 결과의 통계적 평균이라는 데이터 사이언스 기법의 본질에 집중했습니다.

### 4.2 전처리 및 간결한 피처 엔지니어링 (Feature Engineering)

도메인 기반 변수 설계의 부재를 양과 깊이로 밀어붙이는 전략을 차용합니다.

#### A. Head-to-Head (IDTeams) Groupby의 극대화
- 남성부와 여성부의 모든 경기를 전부 병합시킨 `all_games` DataFrame에서, 항시 ID가 작은 팀을 `Team1`, 큰 팀을 `Team2`로 배치하여 두 팀 간의 고유 조합 식별자 `IDTeams`를 생성합니다. (Ex: `1101_1102`)
- 이렇게 수집된 모든 Regular / Tourney 경기에 대해 27종의 경기 세부 스탯(FGM, FGA, OR, DR, Ast, TO, Stl, Blk 등)을 추출합니다.
- 추출한 27개 칼럼에 대해 8개 판다스 통계함수(`sum`, `mean`, `median`, `max`, `min`, `std`, `skew`, `nunique`)를 다중 병렬 적용(`agg()`)시켜 **216개에 달하는 방대한 H2H 매핑 테이블**을 만들어냅니다.
- 추후 이 H2H 스탯들을 토너먼트 전적 로그와 Submit 로그에 결합(`left merge`)하여 독립 변수로 씁니다. (결측치 발생 시 `-1`로 강제 Fillna 처리)

#### B. 타겟 정규화 (점수차 마진)
분류 타겟 예측에 쓰이는 Target은 `Team1`의 승리 여부(`Pred`, 1 or 0) 입니다. 하지만 모델의 통계적 감각을 올려주기 위해 `ScoreDiffNorm`이라는 피처를 하나 더 만듭니다.
```python
self.games['ScoreDiffNorm'] = self.games.apply(
    lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1
)
```
- A팀 이김(10점 차) -> 10
- B팀 이김(10점 차) -> -10
단순 승패 이진 분류(0, 1)의 한계를 보완해줄 수 있는 스코어 피처입니다.

### 4.3 데이터 처리 전략 요약 (비교)

| 항목 | Notebook 4 (본 문서) | 앞선 OTHERS (baseline 등) |
|------|------------------------|-------------|
| **도메인 지표 (Elo 등)** | **완전 0개 (미사용)** | 적극 사용 ($Elo, KP$ 랭킹) |
| **H2H 스탯** | 역대 모든 경기 기준 (Season 무시) | 시즌 혹은 H2H 기준 |
| **누수 통제 (Leakage)** | 고려 안함 (`.groupby()` 전체 사용) | OOF 등 소거법 (일부 사용) |
| **결측치 제어** | 단순 Imputer(`strategy='mean'`) 및 `-1` Fillna | 0 치환 또는 Elo 베이스(1500) |

### 4.4 랜덤 포레스트 구조 및 등방성 예측 확률 교정

이 노트북은 분류(Classification) 시스템 대신 **Random Forest Regressor (연속 회귀)**를 사용하여 오차율을 낮춥니다. 보통 예측 클래스가 극단적으로 쏠리는(0.01 혹은 0.99) Classifier의 단점을 상쇄해 줍니다. 

```python
self.model = RandomForestRegressor(
    n_estimators=235,       # 트리의 숫자를 235개로 강제 고정
    random_state=42, 
    max_depth=15,           # Depth 15로 오버피팅 억제 타협점 설계
    min_samples_split=2,
    max_features='auto'     # (최신 sklearn에서는 deprecated 경고 이슈 있음)
)
```

**모델링 프로세스 (Training Step)**
1. `SimpleImputer`와 `StandardScaler`를 차례로 파이프라인으로 관통시킵니다.
2. 회귀 모델의 훈련 값(`preds`)이 `0.0`~`1.0` 사이의 연속 실수 분포로 산출됩니다. 이를 `np.clip(0.001, 0.999)`로 임의 극단치 차단을 먹여줍니다. Brier Penalty의 폭주시스템을 방지하기 위함입니다.
3. **Isotonic Regression (Train Target 기반 Calibration):** 역시 검증 데이터가 아닌 훈련 셋의 추론값을 다시 훈련 셋의 타겟으로 피팅하여 확률 보정을 수행합니다. 
```python
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(pred, y) 
pred_cal = ir.transform(pred)
```
- 최종적으로 OOF CV 없이 `-cv_scores.mean()`으로 자체 교차 검증 MSE 스코어를 체크한 후(대략 `0.1885`), 보정 커브를 Submission 셋의 확률 변수에 씌워 제출합니다.

### 4.5 종합 평가 코멘트 (Review)

**장점 (Strengths):**
- **강박적인 구조화와 유지보수성:** 코드가 `class` 구조 단 하나로 압축되어 있어, 남녀 구분을 둔다거나 폴드를 이중 삼중으로 파생시키는 여타 스파게티 코드 스크립트 대비 압도적인 깔끔함을 자랑합니다. 데이터셋만 바꿔 끼우면 2026, 2027 시즌에도 1원칙 베이스라인으로 돌릴 수 있습니다.
- **도메인 지식 의존도 탈피:** 머리에 쥐가 나는 농구 통계 지식(포제션 산출 등) 없이도 Pandas Dataframe Aggregation(`max, min, std...`) 머신러닝의 힘만으로 상위 5% 안에 드는 모델 성능을 재증명했습니다.

**숨겨진 리스크 및 한계점 (Limitations):**
- **RandomForestRegressor 하이퍼 파라미터 노후화:** 트리를 235개로 고정하고, `max_features='auto'`를 사용했는데 이는 `sqrt` 보다 훨씬 많은 연산 부하를 초래하며 SKLearn 1.1버전 이상에서 권장하지 않는 옵션입니다. 더 가볍게 최적화 가능한 XGBoost 대비 하드웨어 효율이 많이 떨어집니다.

---

## 5. ncaa-basketball-predictions-with-xgboost.ipynb (2025)

**주요 특징:** 대회에서 흔하게 사용되는 H2H(Head-to-Head) 통계량 집계 방식의 정수를 보여주면서도, 코드를 절차지향적으로 간결하게 짠 스크립트입니다. 4번 노트북과 아이디어를 대다수 공유하며, 외부 지표(Massey 랭킹, Elo 레이팅, 모멘텀 등) 없이 **단일 XGBoost 회귀 모델(XGBRegressor)** 요소 하나만을 사용하여 216개의 방대한 파생 피처를 학습시킵니다. 특히, 5000개의 깊은 부스팅 트리를 GPU로 연산하는 확실한 접근 방식을 택했습니다. 하지만 내부적으로 엄청난 **과적합(Overfitting)**과 Data Leakage(누수)의 흔적을 담고 있기도 한 모델입니다.

### 5.1 사용 데이터

| 파일명 | M/W 병합 패턴 | 용도 |
|--------|--------------|------|
| `*Teams` / `*TeamSpellings` | `pd.concat` 후 조인 | `TeamNameCount` (팀 명칭 변경 빈도) 계산 |
| `*RegularSeasonDetailedResults` | `pd.concat` (`ST='S'`) | H2H 박스 스코어 통계량 집계용 기초 데이터 |
| `*NCAATourneyDetailedResults` | `pd.concat` (`ST='T'`) | H2H 집계용 및 **최종 학습 Target 데이터** 분리 |
| `*NCAATourneySeeds` | 공통 Dictionary 파싱 | 두 팀간 시드 차이(`SeedDiff`) 변수 생성 |
| `SampleSubmissionStage1.csv` | 타겟 프레임 로딩 | Stage 1 평가 매치업 예측용 |

> **데이터 전략 핵심:** M(남)과 W(여) 데이터를 구분 짓지 않고 수직으로 결합(`concat`) 하였으며, Season과 Tourney 기록까지 전체를 결합해(`games`) 최대한 거대한 박스 스코어 이력을 형성합니다.

### 5.2 전처리 및 특성 공학 (Feature Engineering)

도메인 기반 변수나 농구 역학 지표(eFG 등)의 계산은 하나도 없이, 오직 순수한 판다스(Pandas) 그룹화(aggregation) 연산에 의지합니다.

#### A. Head-to-Head(H2H) 글로벌 집계 테이블 생성
안정적인 키 매핑을 위해 더 작은 TeamID를 `Team1`, 더 큰 것을 `Team2`로 고정하고 고유 조합키 식별자인 `IDTeams` (예: `1101_1102`) 를 만듭니다.
```python
c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
```
- 총 27개의 상세 팀 박스 스코어 칼럼에 대해 8개의 통계함수를 동시 적용합니다.
- 이 짧은 연산으로 **216개(27 $\times$ 8)**의 거대한 H2H 파생 피처 테이블 `gb`가 탄생하며, 이를 원본 데이터에 `left_merge` 로 결합합니다.

#### B. 타겟 정규화 (점수차 마진)
단순한 1/0 분류 타겟을 회피하여 회귀 모델(Regressor)의 연속적인 지도학습을 유도하는 기법입니다.
```python
games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
```

#### C. 결측치 제어와 스케일링
- H2H 전적이 없는(단 한 번도 만난 적 없는) 팀 페어의 피처들은 `NaN`으로 남기 때문에 먼저 `-1` 로 강제 채움 처리를 합니다.
- ત્યારબાદ `SimpleImputer(strategy='mean')`와 `StandardScaler()`를 연속 적용시켜 216개 피처 값들의 단위를 스케일링(`X_scaled`)합니다.

### 5.3 학습 데이터 구성

> **학습 데이터 도메인 제어**

```python
games = games[games['ST']=='T']
```
- 앞서 피처를 깎기 위해서 정규시즌(`S`)과 본선 대회(`T`) 데이터를 모두 합쳤었지만, 모델이 실제로 훈련하는 레이블 도메인은 **오로지 토너먼트 본선 데이터(`T`)만**을 남겨 정제했습니다. 정규시즌 양상에 트리가 잡아먹히는 도메인 불일치 현상(Domain Shift)을 원천 차단하기 위함입니다.

### 5.4 모델 학습 (XGBoost 회귀 앙상블 단독 투입)

| 파라미터 | 세팅 값 | 의미 |
|----------|--------|------|
| `n_estimators` | **5000** | 극한의 부스팅 트리 개수로 깊은 곳에 있는 복합 패턴 추출 |
| `learning_rate` | 0.03 | 상대적으로 낮은 보폭으로 과적합 방어 |
| `max_depth` | 6 | 개별 트리의 통제 |
| `device` | **"gpu"** | 5000개 트리를 연산하기 위한 하드웨어 가속기 명시 |
| `random_state` | 42 | 결과 재현성 고정 |

```python
xgb = XGBRegressor(n_estimators=5000, device="gpu", learning_rate=0.03, max_depth=6, random_state=42)
xgb.fit(X_scaled, games['Pred']) 
# ⚠️ Train/Hold-out 분리 없이 전 데이터를 한 번에 학습시키는 치명적 실수 존재!
pred = xgb.predict(X_scaled).clip(0.001, 0.999)
```

### 5.5 완벽한 과적합(Overfitting)과 성능 결과 비교

작성자가 노트북 상에서 출력한 모델 자체 평가 스코어와, 객관적 방어력을 측정한 CV 스코어 사이에 극단적인 간극이 있습니다.

| 종류 | 평가 지표 방식 | 점수 |
|------|-----------|------|
| **Train Set 자기 복제 검증** | Log Loss | `0.0016` |
| **Train Set 자기 복제 검증** | Brier Score | **`0.0000058` ($5.8 \times 10^{-6}$)** |
| **K-Fold 방어 검증** | Cross-validated MSE (5-fold) | **`0.2039`** |

> **분석:** 전체 `games` 데이터를 쪼개지 않고 `fit()` 한 뒤 자신의 Train 데이터에 다시 `predict()` 하였기에 발생한 **비정상적이고 사기적인 Brier Score(0.0000058)** 입니다. 하지만 노트북 마지막단에서 교차 검증용으로 추가된 CV 5-fold 코드를 통과시켜 나온 MSE 값은 **0.2039**로, 베이스라인 노트북 연합(0.15~0.16) 대비 상당히 참혹하게 부서진 실질 방어력을 기록했습니다. 

### 5.6 확률 보정 (Calibration 없음)

Isotonic Regression 같은 전통적인 분포 캘리브레이션은 전부 생략하고, 파이썬 넘파이 빌트인인 확률 차단만 진행했습니다.
- `pred = xgb.predict(...).clip(0.001, 0.999)`
- 예측 분포가 0과 1 양 극단으로 과도하게 쏠리는 부스팅 모델의 특성을 고려할 때, Stage 2에서는 이 클리핑만으로 페널티 폭탄을 상쇄하기엔 무리가 따릅니다.

### 5.7 주요 설계 특징 요약 테이블

| 분석 포인트 | 5번 모델 (ncaa-xgboost) | 다른 우승권 메타 (baseline 등) |
|------|------------------------|-------------|
| **학습 사용 모델** | **XGBRegressor (단일 GPU 가동)** | $XGB + LGBM + Cat$ Stacked 앙상블 |
| **타겟 도메인** | 순수 토너먼트 본선 경기 (`T`) | 토너먼트 + 정규시즌 15% 샘플 레이크 |
| **CV Validation** | **분리 안함 (매우 심각한 Train Overfitting)** | 연도별/Fold별 명확한 CV 구축 |
| **보정 로직** | **단순 클리핑 (`.clip`)** | $Isotonic Regression$ |
| **주입된 피처** | H2H Aggregation (순수 통계 분포) | $Elo, Momentum, Massey$ 등 하이브리드 |
| **모델 연산 속도** | GPU 기반으로 극히 짧은 소모 시간(수십초) | CPU 기반 다중 추론(십수분) |

### 5.8 종합 코멘트 (Review)

**배울 점 (Strengths):**
- **GPU 가속 최적화와 회귀 앙상블:** 케글/구글 코랩 환경에서 어떻게 무거운 트리(5,000개)가 타는 병목을 뚫는지(GPU 디바이스 선언 등) 보여주는 우수한 단일 스크립트 레퍼런스입니다. 확률 분류기(Classifier) 대신 점수 마진(Regressor) 모델을 통과시켜 1, 0의 노이즈를 피해 간 방식이 좋습니다.
- **판다스 집계 극대화의 미학:** 복잡한 도메인 지식 없이 27개의 승소 리포트를 단골 매소드인 `groupby.agg()`로 빠르게 216개 차원으로 폭발시키는 방법론이 매우 직관적입니다.

**주의해야 할 한계점 (Limitations):**
- **대재앙 수준의 Data Leakage (미래 누수):** 4번 노트북이 가졌던 버그를 똑같이 답습합니다. 과거의 특정 연도 매치를 훈련하면서, **미래의 H2H 정보**(10년치 역대 모든 경기를 뭉쳐서 사전 평균 냈으므로)를 무의식적으로 통계 변수에 포함시켜 버린 명백한 컨닝(Look-ahead Bias) 상태입니다.
- **검증 세트 부재와 타겟 과적합:** 본인이 정답을 이미 외워버린 `Train Set`에 예측을 들이붓고 나온 비정상적인 점수($5.8 \times 10^{-6}$)를 도출합니다. 실제 CV 5-fold 평가지표는 무려 MSE 0.203 점 수준으로 박살이 났습니다. 실제 대회에서 이 모델 단독으로는 상위권을 보장할 수 없습니다.
- **분포 보정(Calibration) 없음:** 이진 확률 대회(Brier Score 경연)에서 극단값을 통제하는 분포 형태의 Isotonic 교정을 빼버려서 안전 마진이 없습니다. 업셋(Upset) 단 한 경기에 점수가 곤두박질칠 수 있습니다.
