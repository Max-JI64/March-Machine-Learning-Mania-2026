# 데이터 전처리 파이프라인

## `march-ml-mania-2026-lgbm-xgb-catboost.ipynb` 예시 데이터에서의 전처리
### 사용된 원본 데이터
* **팀 정보**: `MTeams.csv`, `WTeams.csv`
* **시드 번호**: `MNCAATourneySeeds.csv`, `WNCAATourneySeeds.csv`
* **토너먼트 결과**: `MNCAATourneyCompactResults.csv`, `WNCAATourneyCompactResults.csv`
* **정규 시즌 결과**: `MRegularSeasonCompactResults.csv`, `WRegularSeasonCompactResults.csv`
* **매시 순위 (Massey Ordinals)**: `MMasseyOrdinals.csv` (남자 농구 대회 전용)

기존 베이스라인 노트북에서 추출하던 핵심 피처들 역시 별도의 CSV 파일로 모듈화하여 저장합니다.

| 전처리 파트 (분야) | 생성될 CSV 파일명 | 저장될 주요 변수명 (Columns) |
| :--- | :--- | :--- |
| **A. 시드 (Seed) 전처리** | `base_seeds_features.csv` | `Season`, `TeamID`, `SeedNum` |
| **B. 정규 시즌 통계 (Season Stats)** | `base_season_stats_features.csv` | `Season`, `TeamID`, `Games`, `Wins`, `WinRate`, `AvgScore`, `AvgMargin` |
| **C. 매시 순위 보정 (Massey)** | `base_massey_features.csv` | `Season`, `TeamID`, `MOR` (남자팀 대상) |
| **D/E. 매치업 병합 피처** | `base_matchup_features.csv` | `Season`, `T1`, `T2`, `T1_Seed`, `T2_Seed`, `T1_WinRate`, `T2_WinRate` ... `SeedDiff`, `WinRateDiff`, `MarginDiff`, `ScoreDiff`, `MORDiff`, `Label` |

#### 파트별 생성 변수 세부 설명
**A. 시드 (Seed) 전처리**
* 토너먼트 시드 데이터의 문자열(예: 'W01')에서 숫자 부분만 추출하여 순수 시드 번호(`SeedNum`) 피처를 생성.

**B. 정규 시즌 통계 집계 (Season Stats)**
정규 시즌 경기 결과를 바탕으로 시즌별/팀별 집계 데이터를 생성:
* `Games`: 총 경기 수
* `Wins`: 총 승리 수
* `WinRate`: 승률 (`Wins` / `Games`)
* `AvgScore`: 경기당 평균 득점
* `AvgMargin`: 경기당 평균 득실차 (`득점 - 실점`의 평균)

**C. 매시 순위 보정 (Massey Ordinals)**
* 수많은 평가 시스템 중 `SystemName == 'MOR'`(Moore 시스템) 값만 필터링.
* 시즌 및 팀별로 랭킹 산정일(`RankingDayNum`) 기준 가장 **마지막 날짜(최신)**의 순위만 가져와 `MOR` 피처로 사용합니다.

**D. 매치업 피처 구성 및 결측치 처리 (Matchup Features)**
경기할 두 팀의 ID를 오름차순 정렬하여 **T1(Team 1)** 과 **T2(Team 2)** 로 지정 (T1 < T2) 후 통계 지표 차이(Diff) 계산:
* `SeedDiff`, `WinRateDiff`, `MarginDiff`, `ScoreDiff`, `MORDiff` 등 생성.
* 매치업 시드가 없는 경우 `-1`로 결측치 채움.
---
## 새로운 학습용 전처리

각 전처리 파트 분야별로 데이터 모듈화를 위해 개별 CSV 파일로 추출하여 관리할 예정입니다. 아래는 각 파생변수 그룹(분야)이 저장될 CSV 파일 목록 및 변수명입니다.

| 전처리 파트 (분야) | 생성될 CSV 파일명 | 저장될 주요 변수명 (Columns) |
| :--- | :--- | :--- |
| **A. 세부 박스스코어 기반 고도화 지표** | `advanced_stats_features_[M/W].csv` | `Season`, `TeamID`, `FGA`, `FGM`, `OR`, `DR`, `Ast`, `Stl`, `Blk`, `Possessions`, `eFG%`, `TS%`, `OR%`, `TOV%`, `FTRate`, `3PAr`, `OffRtg`, `DefRtg`, `AstTO_Ratio` |
| **B. 후반기 기세 및 흐름 지표** | `momentum_form_features_[M/W].csv` | `Season`, `TeamID`, `Last14_WinRate`, `Last30_WinRate`, `Last14_Margin`, `Last30_eFG%`, `Last30_TOV%`, `Active_WinStreak`, `Last30_QualityWins`, `Elo_Rating_Diff` |
| **C. 농구단 명성 및 누적 기세** | `program_pedigree_features_[M/W].csv` | `Season`, `TeamID`, `3Yr_Rank_Avg`, `Rank_Trend_Slope`, `Tourney_Exp_Score`, `Coach_Tenure`, `Coach_Career_WinRate`, `Coach_Tourney_Wins`, `Is_Rookie_Coach`, `Expected_Seed`, `Power_Composite`, `CoachTWR_Diff` |
| **D. 컨퍼런스 및 스케쥴 강도 보정** | `sos_features_[M/W].csv` | `Season`, `TeamID`, `WinRate_vs_Top50`, `Non_Conf_SOS`, `Is_Major_Conf`, `Major_Conf_WinRate`, `Conf_Tourney_Wins`, `SRS_Diff`, `MasMean_Diff` |
| **E. Brier Score 타겟팅 지표** | `brier_score_features_[M/W].csv` | `Season`, `TeamID`, `Margin_Std`, `Close_Game_WinRate`, `Blowout_Wins_Count`, `Score_Max`, `Score_Std`, `Pyth_Diff` |
| **F. 과거 토너먼트 및 하위 대회 경험** | `past_tourney_features_[M/W].csv` | `Season`, `TeamID`, `Past_Tourney_eFG%`, `Past_Tourney_TOV%`, `Prev_Secondary_Tourney_Success`, `H2H_WinRate`, `TourneyApps_Diff` |
| **G. 일정 및 대진표 기반 휴식 지표** | `schedule_rest_features_[M/W].csv` | `Season`, `TeamID`, `Days_Since_Last_Game` |
| **H. 지리 정보 및 누적 피로도 지표** | `geography_travel_features_[M/W].csv` | `Season`, `TeamID`, `Lat`, `Lon`, `Home_Altitude`, `LateSeason_Altitude_Fatigue`, `LateSeason_Lon_Fatigue` |
| **I. 데이터 증강 (Data Augmentation)** | *(별도 CSV 저장 없이 훈련 시 메모리 호출)* | `Is_Augmented`, 노이즈 추가된 `Diff` 스탯 등 |
> **📌 [중요] 타겟 시즌(Target Season)과 제출 데이터의 개념 정리**
> *   **학습용(Train) 데이터**: 2003년 ~ 2025년까지의 **과거 정규시즌 성적**과 **실제 토너먼트 경기 결과**를 맵핑한 데이터입니다. (예: 2018년 A팀 성적과 B팀 성적을 비교해서, 실제 2018년 토너먼트에서 누가 이겼는지 학습)
> *   **제출용(Test) 데이터**: 현재 진행 중이거나 곧 시작될 **올해(예: 2026년)**의 토너먼트 전 경기 가상 대진표입니다.
> *   즉, 변수를 만들 때는 "예측하려는 타겟 연도(Ex. 2026년)"의 정규시즌 코트(현재 뛰고 있는 선수들) 기록만 뽑아서 써야 합니다. 2026년 결과를 예측하는데 2025년(직전 시즌)의 선수단 스탯을 가져와 섞으면 안 됩니다!

### A. 세부 박스스코어 기반 고도화 지표 (Advanced Stats)
**사용 데이터:** `MRegularSeasonDetailedResults.csv`, `WRegularSeasonDetailedResults.csv`
이 파일들에 있는 승/패 팀별 세부 스탯 컬럼(예: `WFGM`, `LFGM`, `WFGA`, `WOR`, `WTO` 등)을 한 팀 시점으로(W/L 무관하게) 재배열합니다.

**💡 기세 분할 추이 반영 (Rolling Metrics 선호)**:
**시즌 전체 평균(Full Season Average)**만 내면 초반의 모습과 후반의 폼이 섞여버립니다. 이를 보완하기 위해 전체 평균 피처를 유지하되, 추가적인 시계열 분할 지표를 만듭니다.
*   **추천 방식**: Q1~Q4처럼 임의로 4등분(고정된 분기)하는 것보다, **최근 14일, 30일 단위의 롤링(Sliding Window) 평균**을 구하는 것이 모델 성능에 훨씬 유리합니다. 정규시즌 마지막 30일(DayNum 103~132)의 지표가 토너먼트의 현재 '폼(Form)'을 가장 정확히 대변하기 때문입니다. (예: `Last30_eFG%_Diff`, `Last14_Possessions_Diff` 등)
*(참고: 이 세부 스탯들을 "이전 시즌(1년 전, 2년 전)"까지 확장하는 것은 추천하지 않습니다. 대학 농구는 해마다 주전 선수가 2~3명씩 바뀌므로, 2년 전 선배들이 세웠던 3점 슛 성공률이나 리바운드 기록은 올해 후배들의 경기력을 예측하는 데 오히려 방해가(Noise) 될 수 있습니다.)*

*   **기본 세부 스탯 마진 (Raw Stat Diff)**:
    *   **원본 변수:** `FGA`(야투 시도), `FGM`(야투 성공), `OR`(공격 리바운드), `DR`(수비 리바운드), `Ast`(어시스트), `Stl`(스틸), `Blk`(블록)
    *   **공식 및 전처리:** 단순히 복잡한 비율(eFG% 등)만 계산하는 것이 아니라, 두 팀의 1차원적인 스탯 차이(`T1_FGA - T2_FGA`, `T1_OR - T2_OR` 등) 자체도 모델에 그대로 투입하여 트리 모델(LGBM/XGB)이 직접 스탯의 무게감을 판단하도록 합니다.
    *   **최종 생성 변수명:** `FGA_Diff`, `FGM_Diff`, `OR_Diff`, `DR_Diff`, `Ast_Diff`, `Stl_Diff`, `Blk_Diff` (Full 및 Q1~Q4 기간별 생성)

*   **포제션(Possessions) 마진 차이(Diff)**:
    *   **원본 변수:** `FGA`(야투 시도), `OR`(공격 리바운드), `TO`(턴오버), `FTA`(자유투 시도)
    *   **공식:** 팀별 경기당 `Possessions = FGA - OR + TO + (0.475 * FTA)`
    *   **전처리:** 두 팀(T1, T2)의 정규시즌 평균 Possessions를 구한 뒤 `T1_Possessions - T2_Possessions`
    *   **최종 생성 변수명:** `Possessions_Diff`
*   **슈팅 효율성 차이 (eFG% 및 TS% Diff)**: 
    *   **원본 변수:** `FGM`(야투 성공), `FGM3`(3점 성공), `FGA`(야투 시도), `FTA`(자유투 시도), `Score`(득점)
    *   **공식:** 
        *   유효 필드골 성공률(eFG%) = `(FGM + 0.5 * FGM3) / FGA`
        *   트루 슈팅(TS%) = `Score / (2 * (FGA + 0.475 * FTA))`
    *   **전처리:** T1, T2 각각의 시즌 평균 비율을 구하고 Diff 계산.
    *   **최종 생성 변수명:** `eFG%_Diff`, `TS%_Diff`
*   **공격 리바운드 점유율 차이 (OR%)**: 
    *   **원본 변수:** 팀의 `OR`(공격 리바운드), 동일 경기 내 상대팀의 `DR`(수비 리바운드)
    *   **공식:** `OR / (OR + Opp_DR)`
    *   **전처리:** T1, T2 각각의 전체 경기 합산 리바운드 비율을 구하고 Diff 산출.
    *   **최종 생성 변수명:** `OR%_Diff`
*   **턴오버 비율 (Turnover Percentage - TOV%)**:
    *   **원본 변수:** `TO`(턴오버), `Possessions`(포제션)
    *   **공식:** `TO / Possessions`
    *   **전처리:** 해당 비율이 낮을수록 볼 핸들링이 안정적임을 의미. T1과 T2의 TOV% Diff 산출.
    *   **최종 생성 변수명:** `TOV%_Diff`
*   **자유투 획득 비율 (Free Throw Rate - FTRate)**:
    *   **원본 변수:** `FTA`(자유투 시도), `FGA`(야투 시도)
    *   **공식:** `FTA / FGA`
    *   **전처리:** 야투율이 저조한 날이라도 파울을 얻어내 득점력을 유지하는 "끈적임" 지표.
    *   **최종 생성 변수명:** `FTRate_Diff`
*   **3점 슛 의존도 (3-Point Attempt Rate - 3PAr)**:
    *   **원본 변수:** `FGA3`(3점 슛 시도), `FGA`(전체 야투 시도)
    *   **공식:** `FGA3 / FGA`
    *   **전처리:** 3점 슛 의존도가 높을수록 경기 폭발력이 높지만 '기복(Variance)'도 큼. Brier Score 최적화 시 이변 확률을 고려하는 데 매우 중요한 타겟팅 변수.
    *   **최종 생성 변수명:** `3PAr_Diff`
*   **공수 효율성 지수 (Offensive/Defensive Rating - O_Rtg, D_Rtg)**:
    *   **원본 변수:** `Score`, `Opp_Score`, `Possessions`(포제션)
    *   **공식:** `(Score / Possessions) * 100`, `(Opp_Score / Possessions) * 100`
    *   **전처리:** 100번의 공격 기회 당 몇 점을 넣고 몇 점을 내주었는가? 단순 평균 득실점은 경기 템포(Fast Break 팀 등)에 의해 왜곡되므로, 템포를 통제한 '진짜 득실점 마진'을 구합니다.
    *   **최종 생성 변수명:** `OffRtg_Diff`, `DefRtg_Diff`
*   **어시스트/턴오버 비율 (AST/TO Ratio)**:
    *   **원본 변수:** `Ast`(어시스트), `TO`(턴오버)
    *   **공식:** `Ast / TO`
    *   **전처리:** 플레이메이킹과 패싱 게임의 완성도를 보여줍니다. 이 수치가 높은 팀은 강한 수비 압박(프레스)을 당해도 쉽게 무너지지 않습니다.
    *   **최종 생성 변수명:** `AstTO_Ratio_Diff`

### B. 후반기 기세 및 흐름 지표 (Momentum & Form)
**사용 데이터:** `MRegularSeasonCompactResults.csv`, `WRegularSeasonCompactResults.csv`
대회 제출 요건(토너먼트 직전 시점에 모든 경기 가상 예측 제출)을 만족시키기 위해 정규시즌 마감 시점 지표를 고정값으로 씁니다.

*   **단기 및 중기 롤링 승률 차이 (Rolling WinRate Diff)**: 
    *   **원본 변수:** `DayNum`(경기 일자), `WTeamID`, `LTeamID`
    *   **전처리:** 팀별 데이터프레임을 `DayNum` 기준으로 정렬 후, 기준일(정규시즌 마감일 DayNum 132)로부터 **최근 14일(단기)**, **최근 30일(중기)** 구간을 각각 필터링합니다. 해당 구간 내의 `Wins / Games` 비율을 기간별로 각각 계산합니다. (예: `Last14_WinRate_Diff`, `Last30_WinRate_Diff`)
    *   **최종 생성 변수명:** `Last14_WinRate_Diff`, `Last30_WinRate_Diff`
*   **롤링 스탯 마진 및 세부 지표 차이 (Rolling Stat & Margin Diff)**: 
    *   **원본 변수:** `DayNum`, `Score`, `Opp_Score`, `FGA`, `OR`, `TO` 등
    *   **전처리:** 승률뿐만 아니라, 동일하게 '최근 14일', '최근 30일' 필터를 씌운 뒤 해당 기간 동안의 득실차(`Margin`), 포제션(`Possessions`), 슈팅 효율성(`eFG%`), 턴오버 비율(`TOV%`) 등의 평균을 별도로 산출합니다.
    *   이를 통해 "시즌 전체의 eFG%는 낮지만, 최근 14일 동안의 eFG%는 폭발적으로 상승한 팀"처럼 타임라인 기반의 **폼(Form) 상승 곡선**을 모델이 완벽하게 캡처할 수 있습니다. (예: `Last14_Margin_Diff`, `Last30_eFG%_Diff`, `Last30_TOV%_Diff`)
    *   **최종 생성 변수명:** `Last14_Margin_Diff`, `Last30_eFG%_Diff`, `Last30_TOV%_Diff` 등
*   **현재의 연승/연패 흐름 (Active Win/Loss Streak)**:
    *   **원본 변수:** `DayNum`, `WTeamID`, `LTeamID`
    *   **전처리:** 정규시즌 마감일(DayNum 132)을 기준으로, 해당 팀이 토너먼트에 '진입'할 당시의 **최근 연승/연패 상태**를 확인하여 고정 피처(+수치, -수치)로 사용합니다. *(대회 규정상 토너먼트 시작 전에 모든 가상 매치업 결과를 한 번에 예측해야 하므로, 토너먼트 진행 중의 승패 결과는 모델에 반영하지 않고 오직 "토너먼트 시작 전 마지막 정규시즌 경기 시점"의 연승 흐름 하나만 사용합니다.)*
    *   롤링 승률(WinRate)은 경기 수가 섞여서 나오지만, 이 지표는 "어제 경기를 이겼는가"의 날 것 그대로의 사기를 반영합니다.
    *   **최종 생성 변수명:** `Active_Streak_Diff`
*   **시즌 막판 '질 좋은 승리' 횟수 (Quality Wins in Late Season)**:
    *   **원본 변수:** `DayNum`, `WTeamID`, `MMasseyOrdinals` 순위 데이터 조인 (남자 한정)
    *   **전처리:** 최하위권 팀들만 잡은 거품 승률을 걸러내기 위해, 최근 30일 내에 매시 랭킹 Top 50위 이내 강팀을 이긴 숫자(Count)를 산출합니다.
    *   **최종 생성 변수명:** `Last30_QualityWins_Diff`
*   **랭킹 극복 지수 (Upset Value in Late Season)**:
    *   **원본 변수:** `DayNum`, `WTeamID`, `LTeamID`, `MMasseyOrdinals` 순위 데이터
    *   **전처리:** 단순히 '50위 이내인가?' 범주형 카운트 외에, 추가로 **"승리한 경기 당일 기준, 상대 팀의 랭킹과 우리 팀의 랭킹 차이(`My_Rank - Opp_Rank`)"** 변수를 별개의 피처로 추가합니다. 나보다 랭킹이 훨씬 높은 강팀을 잡아낸 경기일수록 기세(Momentum)에 더 큰 가중치를 부여하기 위함입니다.
    *   **최종 생성 변수명:** `Upset_Value_Diff`
*   **[NEW: 기존 I 파트(자체 레이팅)에서 이동] 세밀한 보정이 들어간 홈-원정 어드밴티지 Elo 레이팅 (Home-Adjusted Advanced Elo)**:
    *   **전처리:** 단순 승패 기반 고정 K-Factor Elo가 아니라, **홈 코트 시에는 점수를 보정(예: +75~100)하여 기대승률수식(`We`)을 통제**하고, 경기의 점수 차(Margin) 기복에 따라 **Elo 변동폭 함수(`K-Factor`)에 가중치를 주는** 고급 피처입니다. 타 하위 포스트시즌(Secondary 투어리먼트) 대회의 전적까지 모두 포함하며, 매 시즌 리셋되는 대신 작년 점수의 평균 75%를 계승하는 평균 회귀(Mean Reversion) 방식을 취해 연속성을 부여합니다.
    *   **최종 생성 변수명:** `Elo_Rating_Diff`, `Elo_WinProb_Neutral`, `T1_EloEndSeason`, `T2_EloEndSeason`, `T1_EloPreTourney`, `T2_EloPreTourney`, `IX_Elo_x_Seed`, `IX_Elo_x_Net`

### C. 농구단 명성 및 누적 기세 (Program Pedigree & Historic Trajectory)
선수진은 매년 바뀌더라도 명문 대학(우수 감독, 리크루팅, 시스템)의 전력이나 누적된 큰 경기 경험치는 현재 성적에 영향을 줍니다. 한 해 반짝 잘하는 팀인지, 뼈대 있는 명문인지를 구분하기 위해 이전 3~5개 시즌에 걸친 시계열 지표를 도입합니다.

*   **최근 3시즌 평균 랭킹 (3-Year Historical Rank Average)**:
    *   **사용 데이터:** `MMasseyOrdinals.csv` (남자 한정)
    *   **원본 변수:** `Season`, `OrdinalRank` (SystemName=='MOR' 등의 최종 랭킹)
    *   **전처리:** 현재 예측하려는 `Season` 대비 `(Season-1)`, `(Season-2)`, `(Season-3)`의 정규시즌 최종일자(`RankingDayNum == 133`) 랭킹을 가져와 이동 평균(Moving Average) 산출.
    *   **최종 생성 변수명:** `3Yr_Rank_Avg_Diff`
*   **다년간 랭킹 상승/하락 추이 (Multi-Year Rank Trend/Trajectory)**:
    *   **전처리:** 단순히 '평균'만 내면 100위➜50위➜15위(무서운 상승세)인 팀과 15위➜50위➜100위(몰락하는 팀)이 겉보기에 같은 팀으로 취급됩니다. 이를 방지하기 위해 **"직전 시즌 대비 랭킹 변화량(`Rank_{T-1} - Rank_{T}`)"** 또는 최근 3년간 랭킹의 기울기(Slope) 변수를 1~2개 추가합니다.
    *   *분석:* (변수가 많아지는 것을 걱정하셨지만) 이 정도의 시계열 변수(1~2개 컬럼 추가)는 20여 년 치의 방대한 학습 데이터(수십만 줄의 매치업)를 가진 트리 모델이 거뜬하게 훈련할 수 있으므로 과적합(Overfitting) 걱정 없이 안심하고 추가하셔도 좋습니다!
    *   **최종 생성 변수명:** `Rank_Trend_Slope_Diff`
*   **과거 5년 내 최대 토너먼트 진출 실적 (Recent Tournament Experience)**:
    *   **사용 데이터:** `MNCAATourneyCompactResults.csv`, `WNCAATourneyCompactResults.csv`
    *   **원본 변수:** `DayNum`, `W/L 팀 ID`
    *   **전처리:** 최근 5년간 16강(Sweet 16, `DayNum >= 143`), 8강(Elite 8), 우승 등의 큰 경기 이력을 얼마나 자주 밟아봤는지를 스코어링화. 큰 무대 압박감을 견디는 "DNA" 지표.
    *   **최종 생성 변수명:** `Tourney_Exp_Score_Diff`
*   **감독 부임 기간 및 시스템 안정성 (Coach Tenure / Stability)**:
    *   **사용 데이터:** `MTeamCoaches.csv` (남자 한정 제공 시)
    *   **원본 변수:** `Season`, `CoachName`
    *   **전처리:** 특정 팀(TeamID)에 현재 감독이 몇 `Season`째 연속으로 재직 중인지를 카운트. 장기 집권 중인 감독의 팀은 이변을 쉽게 허용하지 않거나 안정적인 리크루팅을 유지할 확률이 높습니다.
    *   **최종 생성 변수명:** `Coach_Tenure_Diff`
*   **감독 개인 역량 및 경험치 (Coach Career Pedigree)**:
    *   **전처리:** 해당 감독이 과거 타 팀에서 거둔 성적까지 모두 추적(Grouping by `CoachName`)하여 **통산 승률** 및 **통산 토너먼트 진출/승리 횟수**를 구합니다. 또한 부임 직전 대비 팀 성적 상승 수치(**Coach Impact**)도 계산합니다.
    *   **최종 생성 변수명:** `Coach_Career_WinRate_Diff`, `Coach_Tourney_Wins_Diff`
*   **초보 감독 및 결측치 예외 처리 (Rookie & Missing Handlers)**:
    *   **전처리:** 이번이 커리어 첫 시즌인 초보 감독의 경우 통산 기록을 0으로 처리하되 `Is_Rookie_Coach` 플래그를 추가합니다. `WTeams`(여성부) 등 감독 데이터가 아예 없는 경우는 해당 수치들을 `-1`로 결측 처리하고 `Is_Coach_Missing` 플래그를 부여해 트리 모델(LGBM/XGB)이 예외 그룹으로 식별하여 학습할 수 있도록 유도합니다.
    *   **최종 생성 변수명:** `T1_Is_Rookie`, `T2_Is_Rookie`, `Is_Coach_Missing`
*   **1라운드 이변 특화 DNA (First Round Upset DNA)**:
    *   **전처리:** 특정 대학(Team)이나 감독(Coach)이 과거 토너먼트 **1라운드(Round of 64)**에서 강팀(예: 1~4번 시드)을 상대로 업셋(Upset)을 일으킨 횟수나 승률을 별도로 집계합니다. 토너먼트 전체 경험치(`Tourney_Exp_Score`)와 달리, 이 변수는 "광기의 3월(March Madness)" 초반 특유의 이변 발생 확률을 캡처하는 데 특화됩니다.
    *   **최종 생성 변수명:** `First_Round_Upset_Rate_Diff`
*   **[NEW: 기존 I 파트(자체 레이팅)에서 이동] 파워 랭킹 결합 및 기대 시드 (Power Composite & Expected Seed)**:
    *   **전처리:** 위에서 구한 `Elo`, `SRS`, 그리고 `AdjNetRtg`, `WinPct_recent`, `Margin_recent` 피처들을 모두 Z-Score 정규화로 스케일링한 후, 각각 0.3, 0.2 등의 고정 계수 가중치를 두어 하나로 합칩니다. 이 **'Power Composite'** 점수를 바탕으로 전체 현존하는 68개 팀 내의 가상 등수를 매겨 **자체 기대 시드(Expected Seed)**를 역산해 부여합니다.
    *   **최종 생성 변수명:** `Expected_Seed_Diff`, `Power_Composite_Diff`, `Actual_vs_Expected_Seed_Ratio`, `T1_ExpectedSeed`, `T2_ExpectedSeed`
*   **[NEW: 기존 I 파트(자체 레이팅)에서 이동] 감독 토너먼트 승률 (Coach Tournament Records)**:
    *   **전처리:** 현재 시즌까지 해당 감독(Coach)의 역대 토너먼트 종합 누적 승/패를 1차원적으로 합산 계산하여 토너먼트 무대 승률을 생성합니다 (초보나 정보 부재 감독은 0.5 중립처리).
    *   **최종 생성 변수명:** `CoachTWR_Diff`, `T1_CoachTWR`, `T2_CoachTWR`

### D. 컨퍼런스 및 스케쥴 강도 보정 (Strength of Schedule)
**사용 데이터:** `MMasseyOrdinals.csv` (남자), `MTeamConferences.csv`, `WTeamConferences.csv`

*   **상위권 랭크 팀 상대 승률 (WinRate vs Top 50)**: 
    *   **원본 변수:** `MMasseyOrdinals.csv` 시스템 랭킹 및 `*RegularSeasonCompactResults.csv` 경기 날짜(`DayNum`).
    *   **전처리 (강팀 판독기):** 10승을 했더라도 약팀만 10번 이긴 것과 우승 후보를 10번 이긴 것은 완전히 다릅니다. 이를 식별해내는 과정입니다.
        1.   `MMasseyOrdinals.csv`에 있는 수많은 시스템 중 신뢰도가 가장 높은 컴퓨터 기반 알고리즘(예: `SystemName == 'MOR', 'POM', 'SAG' 등`) 1개를 고정합니다.
        2.   대회 룰(미래 데이터 참조 불가)에 맞추기 위해, **각 팀이 정규시즌 경기를 치를 당시(해당 `DayNum` 기준)** 상대 팀의 가장 최근 순위표를 매핑(`merge_asof` 기법)합니다. (예: 12월 경기에 2월 랭킹을 가져다 쓰면 안 됨)
        3.   이렇게 매핑된 모든 경기들 중에서, 상대방의 당시 랭킹이 **1위 ~ 50위 이내(`OrdinalRank <= 50`)**였던 빡센 경기들만 따로 필터링합니다.
        4.   해당 조건 하에서 치러진 총 경기 수 대비 승리한 횟수를 나누어 **50위권 이내 강팀 상대 전적 승률(`WinRate_vs_Top50`)**을 산출합니다.
    *   **최종 생성 변수명:** `WinVsTop50_Diff` (T1의 % - T2의 % 차이값)
*   **비컨퍼런스 일정 강도 (Non-Conference SOS)**:
    *   **전처리:** 정규시즌 초반(보통 11~12월, `DayNum` 대략 50 이전)에 치러지는 타 컨퍼런스 팀과의 맞대결 데이터를 추출합니다. 이때 맞붙은 상대 팀들의 평균 시즌 최종 랭킹(`MMasseyOrdinals`) 또는 평균 승률을 계산합니다. 진정한 강팀은 약소 컨퍼런스 소속이더라도 시즌 초반에 자발적으로 험난한 원정 스케줄을 짜서 경험치를 쌓습니다.
    *   **최종 생성 변수명:** `Non_Conf_SOS_Diff`
*   **Major 컨퍼런스 가중치 및 승률 보정**: 
    *   **원본 변수:** `*TeamConferences.csv`의 `ConfAbbrev` 컬럼.
    *   **전처리:** 제공된 `Conferences.csv`를 기준으로 흔히 **'Power 6'**라 불리는 6대 메이저 컨퍼런스(`acc`, `big_east`, `big_ten`, `big_twelve`, `pac_twelve`, `sec`) 소속인지를 0 또는 1로 One-Hot 처리합니다. 이들 컨퍼런스는 수준 차이가 극심하므로, 단순히 소속 여부뿐만 아니라 **"해당 6대 컨퍼런스 내부 경기에서의 승률"**을 별도로 계산하여 피처로 빼둡니다. (예: 약소 컨퍼런스 20승 무패 팀과, SEC 컨퍼런스 12승 8패 팀의 가치를 모델이 정확히 비교할 수 있게 함)
    *   **최종 생성 변수명:** `T1_Is_Major_Conf`, `T2_Is_Major_Conf`, `Major_Conf_WinRate_Diff`
*   **컨퍼런스 토너먼트의 광기 (Conference Tourney Performance)**:
    *   **사용 데이터:** `MConferenceTourneyGames.csv`, `WConferenceTourneyGames.csv`
    *   **전처리:** 정규시즌 마감 직후, 본선 진출권을 따내기 위해 치러지는 '컨퍼런스 토너먼트'에서의 폼을 측정합니다. 여기서 결승까지 가거나 우승을 차지한 팀(Automatic Bid)은 그 기세가 이어져 본선(NCAA 토너먼트)에서도 좋은 성적을 낼 확률이 높습니다. '최근 치른 컨퍼런스 토너먼트 승수'를 카운트합니다.
    *   **최종 생성 변수명:** `Conf_Tourney_Wins_Diff`
*   **[NEW: 기존 I 파트(자체 레이팅)에서 이동] SRS (Simple Rating System) 선형대수학 강도 스코어 기반 가짜 스탯 판독기**:
    *   **전처리:** 팀의 '득실 마진 배열'과 해당 팀들이 맞붙은 '상대 전적 네트워크 행렬'을 만들어, 파이썬 NumPy 선형 방정식(`np.linalg.solve`)을 수학적으로 풉니다. 이 방정식을 통해 "약팀 학살로 쌓은 가짜 마진"과 "강팀 원정에서 기록한 진짜 마진"을 분별할 수 있는 핵심 구조적 강도 스코어를 산출합니다.
    *   **최종 생성 변수명:** `SRS_Diff`, `T1_SRS_recent`, `T2_SRS_recent`
*   **[NEW: 기존 I 파트(자체 레이팅)에서 이동] 기계학습/외부 Massey Ordinals 통합 랭킹 지표**:
    *   **전처리:** POM, SAG, MOR, COL 등 7~8개 주요 컴퓨터 랭킹 시스템을 선별하여, 각 시스템별 시즌 마지막 날짜의 `OrdinalRank`를 가져와 0~1 사이의 백분위 또는 Z-스코어로 변환합니다. 변환된 순위의 평균치, 최저치, 중앙값 및 여러 랭킹 시스템 간의 불일치성(일관성)을 잴 수 있는 **표준편차(MasStd)**를 구합니다.
    *   **최종 생성 변수명:** `MasMean_Diff`, `MasMin_Diff`, `MasStd_Diff`, `MasPOM_Diff`, `MasSAG_Diff`, `Massey_x_Elo`

### E. ★ Brier Score 타겟팅 지표 (Probability & Variance) ★
**사용 데이터:** `MRegularSeasonCompactResults.csv`, `WRegularSeasonCompactResults.csv`

*   **득실차 표준편차 (Margin Std Dev)**: 
    *   **원본 변수:** 팀의 경기당 득실차 = (`Score - Opp_Score`).
    *   **전처리:** 정규시즌에 치른 팀의 모든 경기 득실차 배열(Array)에 대해 Pandas `.std()` 적용. 기복을 수치화.
    *   **최종 생성 변수명:** `Margin_Std_Diff`
*   **박빙 및 클러치 능숙도 (Close Game WinRate)**: 
    *   **원본 변수:** 양 팀 점수의 절대값 차이 `abs(WScore - LScore)`.
    *   **전처리:** 위 점수차가 5점 이하인 Close Game만 필터링. 이 경기들에서의 승률 계산. 박빙 승부에 강한 팀은 토너먼트의 숨 막히는 압박감 속에 높은 확률로 살아남습니다.
    *   **최종 생성 변수명:** `Close_Game_WinRate_Diff`
*   **Blowout 승리 및 가비지 타임 비율 (Blowout Win Margin)**:
    *   **원본 변수:** 득실차 = (`WScore - LScore`).
    *   **전처리:** 점수 차가 20점 이상 벌어진 대승(Blowout Win) 경기 횟수나, 대승한 경기의 평균 득실차를 따로 뽑습니다. 압도적인 승리 지표는 모델이 해당 팀의 승리 확률을 0.51 수준이 아닌 0.8~0.9까지 과감하게 쏘아 올리도록 (Brier Score 로스를 줄이도록) 확신을 줍니다.
    *   **최종 생성 변수명:** `Blowout_Wins_Count_Diff`
*   **득점의 기복 및 폭발력 (Score Variance & Max potential)**:
    *   **전처리:** 단순히 팀의 '평균 득점'만 보지 않고, 정규시즌 득점 배열에서 **최대 득점(Max)** 또는 당일 슈팅감에 크게 의존하는 팀인지 확인하기 위해 **득점의 표준편차(Score Std Dev)**를 봅니다. 폭발력이 높은 팀은 강팀을 잡을 가능성도, 약팀에게 잡힐 가능성도 높다는 뜻이므로 Brier Score 예측 시 확률을 0 극단으로 몰아주지 못하게 하는 억제기 역할을 합니다.
    *   **최종 생성 변수명:** `Score_Max_Diff`, `Score_Std_Diff`
*   **[NEW: 기존 I 파트(자체 레이팅)에서 이동] 빌 제임스 피타고리안 기댓값 (Pythagorean Expectation) 및 시드 교호작용**:
    *   **전처리:** 득점 비율(`AvgScore^11.5 / (AvgScore^11.5 + AvgAllow^11.5)`)에서 나오는 순수 승리 확률을 도출합니다. 또한, 자체 레이팅(Elo/Net Rating)과 실제 배치된 '토너먼트 시드' 사이의 차이 혹은 곱(`Interaction`)을 통해, 실제 실력은 좋은데 억울하게 낮은 시드를 받은 "업셋(Upset)" 기대 팀을 모델이 찾아내도록 돕습니다.
    *   **최종 생성 변수명:** `Pyth_Diff`, `IX_Seed_x_Pyth`, `SeedWinProb`, `UpsetScore`, `HotnessScore`

> **🚨 [치명적 주의사항: Data Leakage 및 미래 데이터 참조 금지]**
> *   과거 캐글 대회의 가장 흔한 실수이자, **이 파이프라인에서 절대 추가해서는 안 되는 기획**: "토너먼트 1라운드(64강) 성적을 바탕으로 2라운드(32강) 결과를 예측하는 변수".
> *   **이유:** `README.md(Main)` 규칙에 나오듯 토너먼트 시작 전 모든 가능한 가상 대진표(예: 1번 시드 vs 16번 시드, 1번 시드 vs 8번 시드 등)에 대한 예측을 **미리 100% 한 번에 제출**해야 합니다.
> *   따라서, 실제 2026년 토너먼트가 개막한 뒤 특정 팀이 1라운드에서 압도적으로 이겼다고(기세가 올랐다고) 하더라도, 그 `어제 경기(토너먼트 1R) 결과값`을 오늘 2라운드 예측 변수로 사용할 수 없습니다.
> *   **결론:** 우리의 모든 롤링 데이터 및 기세(Momentum) 변수 계산의 **타임라인 마지노선은 반드시 `정규시즌 마지막 날(DayNum=132)`**에서 끊겨야 하며, 예측 대상인 토너먼트 경기(`DayNum >= 134`)의 데이터는 모델 추론 시점에 절대 결합되어서는 안 됩니다.

### F. 과거 토너먼트 및 하위 대회 경험 (Historical & Secondary Tourney Exp)
**사용 데이터:** `*NCAATourneyDetailedResults.csv`, `*SecondaryTourney*.csv`
정규시즌이 아닌 과거 토너먼트 및 포스트시즌(NIT, CBI 등)에서의 성과와 세부 스탯을 바탕으로 "큰 경기 DNA"와 "다크호스 잠재력"을 수치화합니다.

*   **과거 토너먼트 슈팅 효율성 및 안정성 (Historical Tourney Clutch Form)**:
    *   **원본 변수:** 과거 시즌 토너먼트 경기들의 `FGM`, `FGA`, `FGM3`, `TO`, `Possessions` 등
    *   **전처리:** 특정 팀의 과거 토너먼트 경기들만 모아 `eFG%`, `TOV%` 등을 재계산합니다. 큰 무대 압박감 속에서 유독 야투율이 떨어지거나 턴오버가 급증하는 팀(또는 반대로 집중력이 올라가는 팀)을 식별하는 DNA 지표입니다.
    *   **최종 생성 변수명:** `Past_Tourney_eFG%_Diff`, `Past_Tourney_TOV%_Diff`
*   **작년 하위 대회 성과를 통한 다크호스 감별 (Prev Secondary Tourney Success)**:
    *   **원본 변수:** 직전 1~2개 시즌의 `*SecondaryTourneyCompactResults.csv`
    *   **전처리:** 작년에 NCAA 메인 토너먼트에 진출하지 못해 NIT, CBI 등 하위 포스트시즌 대회에 나갔던 팀 중 우승이나 4강 등 딥런(Deep Run)을 기록한 경우 가중치를 부여합니다. 이러한 팀들은 다음 해 메인 토너먼트에서 이변을 일으키는 경우가 잦습니다.
    *   **최종 생성 변수명:** `Prev_Secondary_Tourney_Success_Diff`
*   **[NEW: 기존 I 파트(자체 레이팅)에서 이동] H2H 상대 전적 확률 축소 (Bayesian Head-to-Head)**:
    *   **전처리:** T1과 T2 간 역대 포스트시즌(토너먼트) 맞대결이 존재한다면 과거 승률을 산출하되, 표본이 1~2개 밖에 없는 경우의 극단값 과적합을 방지하기 위해 `사전 확률(0.5)`과 가상 3경기 분량을 섞어 베이지안 축소 변수를 생성합니다.
    *   **최종 생성 변수명:** `H2H_WinRate`, `H2H_Games`
*   **[NEW: 기존 I 파트(자체 레이팅)에서 이동] 토너먼트 경험 6년 누적 지표 (Tournament Experience)**:
    *   **전처리:** 최근 6개 연도 동안 각 팀이 토너먼트에 얼마나 자주 출전했는지, 평균 몇 승을 올렸는지, 16강 이상의 딥런(`DeepRunRate`)을 했는지를 누적 계산하여 수치화시킵니다.
    *   **최종 생성 변수명:** `TourneyApps_Diff`, `TourneyWPG_Diff`, `DeepRunRate_Diff`, `SeedAvg_Diff`

### G. 일정 및 대진표 기반 휴식 지표 (Schedule & Rest Advantage)
**사용 데이터:** `*Seasons.csv`, 경기 날짜 정보(`DayNum`)
대진표 구성 원리와 날짜 계산을 바탕으로 팀의 피로도 회복과 체력적 우위를 측정합니다.

*   **토너먼트 직전 순수 휴식일 (Rest Days Advantage)**:
    *   **원본 변수:** `Seasons.csv`의 `DayZero`, 각 정규시즌/컨퍼런스 토너먼트의 마지막 경기 `DayNum`
    *   **전처리:** 정규시즌 종료 혹은 컨퍼런스 토너먼트 탈락 날짜부터 NCAA 토너먼트 1라운드 시작일 사이에 **며칠을 쉬었는지**(`Days_Since_Last_Game`)를 계산합니다. 체력 회복과 경기 감각 저하 사이의 밸런스를 측정하는 지표로 활용할 수 있습니다.
    *   **최종 생성 변수명:** `Days_Since_Last_Game_Diff`

### H. 지리 정보 및 누적 피로도 지표 (Geography & Travel)
**사용 데이터:** `Cities.csv`, `*GameCities.csv`, 외부 학교별 좌표/고도 데이터

대회 제출 규정 상 어떤 팀끼리 만날지에 대한 "모든 가능한 가상 매치업"을 예측해야 하므로 **해당 가상 토너먼트 경기가 실제로 어느 도시에서, 몇 라운드에 열릴지 테스트 시점에는 알 수 없습니다.** 따라서 토너먼트 당일의 이동 거리나 대진표 난이도보다는, 정규시즌 동안 누적된 피로도와 팀 고유의 지리적 체력 특성을 추출하는 데 집중합니다.

*   **시즌 막판 고도 충격량 (Late Season Altitude Fatigue_Score)**:
    *   **전처리:** 고도의 급격한 변화(오르락내리락)는 비행 거리에 관계없이 선수들에게 엄청난 피로와 멀미를 유발합니다. 정규시즌 마지막 30일 동안 [내 홈구장 ↔ 원정 구장 1 ↔ 원정 구장 2 ↔ 내 홈구장]을 오갈 때 겪은 고도 격차의 절대값(`abs(Alt_diff)`)을 모두 누적 합산합니다.
    *   **최종 생성 변수명:** `LateSeason_Altitude_Fatigue_Diff`
*   **고산지대 생존 경험치 (Season High Altitude Experience)**:
    *   **전처리:** 정규시즌 동안 자신의 평소 홈구장 고도보다 현저히 높은(예: +1000m 이상) 고산지대 원정 구장에서 경기를 치르고 **승리한 경험(Experience / WinRate)**을 수치화합니다. 고지대 산소 부족 환경을 버텨낸 팀은 토너먼트에서 어떤 구장이 배정되어도 환경 변화에 흔들리지 않습니다. (정규시즌엔 `*GameCities.csv`를 통해 경기장 좌표를 알 수 있으므로 모델 타겟팅 가능)
    *   **최종 생성 변수명:** `High_Alt_WinRate_Diff`
*   **순수 홈 고도 격차 및 기후/지역 특성 (Home Geo & Altitude Advantage)**:
    *   **전처리:** 로키산맥 출신 팀과 해수면 출신 팀은 중립이나 원정 등 어디서 만나든 근본적인 심폐 지구력과 체력 회복 속도의 차이가 있습니다. 또한, 대학 농구 특성상 듀크, 캔자스 등 전통적인 강세 지역(Hotspot)의 인프라가 승률에 영향을 줍니다. 따라서 각 팀 캠퍼스의 **고도(Altitude)** 차이뿐만 아니라, **절대 위도(Lat)와 경도(Lon)** 수치 자체를 피처에 그대로 투입하여 트리 모델이 미국 내 지역적 강세를 스스로 파악하게 돕습니다. (단, 경기장 위치를 모르는 대회 특성상 T1과 T2의 위경도 차이(`Diff`)는 시차 계산의 노이즈가 될 수 있어 제외하고 절대 좌표만 넣습니다.)
    *   **최종 생성 변수명:** `Home_Altitude_Diff`, `T1_Lat`, `T1_Lon`, `T2_Lat`, `T2_Lon`
*   **시즌 막판 누적 시차/원정 피로도 (Late Season Lon & Travel Fatigue)**:
    *   **전처리:** 서부와 동부를 오가는 원정 스케줄은 선수들의 생체 리듬(Timezone)을 망가뜨립니다. 마지막 30일(DayNum 103~132) 동안 이동한 **순수 2D 거리**와 함께, 타임존의 변화를 대변하는 **경도 이동량(Longitude Change 절대값)**을 롤링 합산하여 수면 리듬이 깨진 채 토너먼트에 간신히 턱걸이한 팀을 식별합니다.
    *   **최종 생성 변수명:** `LateSeason_Lon_Fatigue_Diff`, `LateSeason_Travel_Fatigue_Diff`

### I. 데이터베이스 증강 기법 (Data Augmentation) - 함수 호출 전용
> 📌 **주의**: I 파트는 팀별 통계 CSV 파일을 생성하는 것이 아닙니다. A~H 파트까지 생성된 피처들을 모아 **최종 학습용 매치업 데이터프레임(`train_df`)을 만들었을 때, 모델 학습 직전에 메모리 상에서만 데이터를 증폭(Augmentation)**하는 훈련 전용 함수 로직입니다. 

**사용 데이터:** 학습용 기반 매치업 풀 세트 (정규시즌/토너먼트/하위 포스트시즌 결과 등 A~H 파트를 통해 완성된 매치업 140~300개 피처의 배열)

*   **1. 데이터 대칭 스왑 증강 (Symmetric Data Swapping Augmentation) 처리**:
    *   **전처리 (Train 데이터 한정):** T1이 T2를 이겼다(Label=1)는 것은 T2가 T1에게 졌다(Label=0)는 완벽한 대칭입니다. Train 데이터에서 T1-T2 관련 열(`T1_` ↔ `T2_` 교체, 차이값인 `_Diff`의 부호 반전 `-Diff`)의 순서를 강제로 모두 바꾼(Flip) 반전 데이터열을 만들고, `Label = 1 - Label`로 만들어 기존 Train Data에 1배수 더 복사(Concat)하여 이어 붙입니다.
    *   **효과:** 이 과정으로 학습 데이터 크기가 정확히 **2배(Augmentation)**로 늘어납니다. 모델이 특정 기준 팀 순번(TeamID의 크고 작음에 따른 배열 오류)에 편향되지 않고 어떤 자리에 가도 완벽하게 좌우 대칭성을 인식해 중립적으로 판정할 수 있도록 유도합니다.
    *   **최종 식별 변수명:** 생성된 데이터 행에는 구분을 위해 `Is_Augmented=1` 피처를 부여할 수 있게 만듭니다.
*   **2. 연속형 데이터 가우시안 노이즈 (Gaussian Noise for Continuous Features)**:
    *   **전처리 (Train 데이터 한정):** 모델의 과적합(Overfitting)을 억제하기 위해 승리에 영향을 미치는 `Possessions_Diff` 나 경기력 마진 관련 연속형 변수 공간에 정규분포 가우시안 노이즈(Gaussian Noise)를 미세하게 더해줍니다 (`NOISE_SCALE = 0.02 ~ 0.03` 혹은 `X + N(0, 0.02)`).
    *   **효과:** 테스트 셋(미래 예측분)과 기존 OOF Validation이 다르게 나올 경우를 고려해 강건성(Robustness)을 부여합니다. 기존 원본 행과 노이즈가 기입된 추가 행으로 데이터셋 덩치를 한 번 더 부풀립니다.
*   **3. Recency 가중치 (Recency Sample Weights 지수 감쇠)**:
    *   **전처리 (Train 데이터 한정):** 오래전 데이터(예: 2008년도 매치업)와 최근 데이터(예: 2025년도)를 완전히 동등한 중요도로 트리 모델이 학습하지 못하게 만듭니다. 학습 대상 년도 역순으로 지수 감쇠(Decay) 계수 `w_tr = 0.60 ^ (max_season - Season)` 공식(혹은 비슷한 타임스탬프 비례)을 사용해 `weight` 값을 설정합니다.
    *   **최종 식별 변수명:** `Sample_Weight` (LGBM, XGBoost 학습 fit 단계에서 sample_weight 파라미터로 넘겨줌)
*   **4. 신경망(NN)용 Label Smoothing 적용 기법**:
    *   **전처리 (Train 및 Stacking 단계):** 딥러닝 앙상블 추가 시 OOF Brier Score에 0과 1 근접 극단치 패널티를 덜 부과받기 위해. 확률을 부드럽게 변환시킵니다.
    *   **공식 결합:** `y_smooth = y * 0.95 + 0.5 * 0.05`

---
**추후 계획**: 위에서 제안한 새로운 파생 변수들의 전처리 로직(공식, 사용 피처 맵핑)을 Python 코드로 구현하여 기존 `T1 vs T2 Diff` 파이프라인에 병합할 예정입니다.
