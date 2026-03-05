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
