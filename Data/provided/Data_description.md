# 데이터셋 설명 (Dataset Description)

본 대회에서 제공된 데이터 파일(`*.csv`)은 과거 팀 데이터, 경기 결과, 랭킹 및 대진표 정보를 포함합니다. 모든 파일은 동일한 형식을 가지며 남성(M) 데이터와 여성(W) 레이블은 `M`과 `W` 접두사로 구분됩니다.

## 데이터 섹션 1 - 기본 사항 (The Basics)

정규 시즌 및 NCAA 토너먼트 경기를 예측하기 위한 기본 데이터입니다. 팀, 시즌, 토너먼트 시드 및 경기 결과에 대한 정보 등이 있습니다.

### MTeams.csv 및 WTeams.csv
각 대학 팀을 식별하는 기본 정보 파일입니다.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example - MTeams) |
| :--- | :--- | :--- |
| **TeamID** | 팀 고유 식별자 (남성 1000 ~ 1999, 여성 3000 ~ 3999) | `1101` |
| **TeamName** | 팀의 대학명 (16자 이하) | `Abilene Chr` |
| **FirstD1Season** | 디비전 I 학교로 처음 등장한 시즌 (남성만 존재) | `2014` |
| **LastD1Season** | 디비전 I 학교로 마지막 등장한 시즌 (남성만 존재) | `2026` |

### MSeasons.csv 및 WSeasons.csv
과거 데이터에 포함된 각 시즌의 지역 설정과 기준 날짜 정보입니다.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example - MSeasons) |
| :--- | :--- | :--- |
| **Season** | 토너먼트가 끝나는 연도 | `1985` |
| **DayZero** | 시즌의 기준 날짜 (DayNum=0) | `10/29/1984` |
| **RegionW, X, Y, Z** | 4개 토너먼트 지역 식별자 | `East`, `West`, `Midwest`, `Southeast` |

### MNCAATourneySeeds.csv 및 WNCAATourneySeeds.csv
각 NCAA 토너먼트 참가 팀의 시드 정보입니다.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example - MNCAATourneySeeds) |
| :--- | :--- | :--- |
| **Season** | 토너먼트가 치러진 연도 | `1985` |
| **Seed** | 지역 및 시드를 식별하는 문자 (예: W01) | `W01` |
| **TeamID** | 팀 식별자 | `1207` |

### MRegularSeasonCompactResults.csv 및 WRegularSeasonCompactResults.csv
정규 시즌 경기(셀렉션 선데이 포함 이전 경기의 합) 기본 결과입니다.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example - MRegularSeason) |
| :--- | :--- | :--- |
| **Season** | 경기가 속한 시즌 연도 | `1985` |
| **DayNum** | DayZero로부터 경과한 일수 | `20` |
| **WTeamID** | 승리한 팀 ID | `1228` |
| **WScore** | 승리한 팀 획득 점수 | `81` |
| **LTeamID** | 패배한 팀 ID | `1328` |
| **LScore** | 패배한 팀 획득 점수 | `64` |
| **WLoc** | 승리 팀의 경기 장소 (`H`: 홈, `A`: 원정, `N`: 중립) | `N` |
| **NumOT** | 연장전 횟수 | `0` |

### MNCAATourneyCompactResults.csv 및 WNCAATourneyCompactResults.csv
NCAA 토너먼트 경기의 기본 결과입니다. 정규 시즌 결과와 형식이 일치합니다.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example - MNCAATourney) |
| :--- | :--- | :--- |
| **Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT** | 정규 시즌 결과 파일과 의미 동일 (WLoc은 남성 항상 N) | `1985, 136, 1116, 63, 1234, 54, N, 0` |

### SampleSubmissionStage1.csv 및 SampleSubmissionStage2.csv
결과 제출 예시 파일입니다.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| **ID** | 시즌과 두 팀의 ID 조합 (`시즌_약팀ID_강팀ID`) | `2022_1101_1102` |
| **Pred** | ID 상의 첫 번째 팀이 이길 예측 확률 | `0.5` |


## 데이터 섹션 2 - 팀 박스 스코어 (Team Box Scores)

2003(남성), 2010(여성) 시즌 이후 팀 단위 상세 경기 통계입니다. CompactResults 열에 상세 결과 열을 더한 구조입니다.

### MRegularSeasonDetailedResults.csv / WRegularSeasonDetailedResults.csv
### MNCAATourneyDetailedResults.csv / WNCAATourneyDetailedResults.csv

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| (첫 8개 열) | Compact Results 파일과 동일 | `2003, 10, 1104, 68...` |
| **WFGM / LFGM** | 야투 성공 횟수 (승/패 팀) | `27` / `22` |
| **WFGA / LFGA** | 야투 시도 횟수 | `58` / `53` |
| **WFGM3 / LFGM3** | 3점슛 성공 횟수 | `3` / `2` |
| **WFGA3 / LFGA3** | 3점슛 시도 횟수 | `14` / `10` |
| **WFTM / LFTM** | 자유투 성공 횟수 | `11` / `16` |
| **WFTA / LFTA** | 자유투 시도 횟수 | `18` / `22` |
| **WOR / LOR** | 공격 리바운드 | `14` / `10` |
| **WDR / LDR** | 수비 리바운드 | `24` / `22` |
| **WAst / LAst** | 어시스트 | `13` / `8` |
| **WTO / LTO** | 실책 (Turnovers) | `23` / `18` |
| **WStl / LStl** | 스틸 (Steals) | `7` / `9` |
| **WBlk / LBlk** | 블록 (Blocks) | `1` / `2` |
| **WPF / LPF** | 개인 파울 | `22` / `20` |


## 데이터 섹션 3 - 지리 정보 (Geography)

2010 시즌 이후 정규 시즌 및 토너먼트 경기가 열린 도시 위치 정보입니다.

### Cities.csv
모든 경기가 열린 도시 마스터 목록입니다. (남녀 구분 없음)

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| **CityID** | 개최 도시 고유 4자리 ID | `4001` |
| **City** | 도시 이름 텍스트 | `Abilene` |
| **State** | 도시에 해당하는 주(State) 약자 (멕시코는 MX) | `TX` |

### MGameCities.csv 및 WGameCities.csv
2010 시즌부터의 각 경기 개최 도시 매핑 데이터입니다.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example - MGameCities) |
| :--- | :--- | :--- |
| **Season, DayNum, WTeamID, LTeamID** | 해당 경기를 식별하는 열 | `2010`, `7`, `1143`, `1293` |
| **CRType** | 경기 유형 (`Regular`, `NCAA`, `Secondary`) | `Regular` |
| **CityID** | 경기가 치러진 도시 ID | `4027` |


## 데이터 섹션 4 - 공개 순위 (Public Rankings)

평가 기관들의 주간 남성 데이터 랭킹 정보 (2003~) 입니다.

### MMasseyOrdinals.csv
서수 순위 데이터 (#1, #2... 순위) 명단입니다. (남성 전용)

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| **Season** | 랭킹 발표 토너먼트 시즌 연도 | `2003` |
| **RankingDayNum** | 랭킹 기준을 사용할 수 있는 일자 기준 | `35` |
| **SystemName** | 순위 예측/평가 기관 약어명 | `SEL` |
| **TeamID** | 평가 팀 식별자 | `1102` |
| **OrdinalRank** | 전체 평가 대상 서수 순위 | `159` |


## 데이터 섹션 5 - 보충 자료 (Supplements)

감독, 컨퍼런스, 철자 구조 및 기타 토너먼트 경기에 관한 부가 정보 테이블입니다.

### MTeamCoaches.csv
각 남성 팀의 감독 시즌별 정보입니다.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| **Season** | 코치 재임 연도 | `1985` |
| **TeamID** | 담당 팀 식별자 | `1102` |
| **FirstDayNum, LastDayNum** | 해당 코치의 단일 시즌 내 담당 일자 범위 | `0`, `154` |
| **CoachName** | 코치의 전체 이름(소문자 형태, 언더바 처리) | `reggie_minton` |

### Conferences.csv
각 컨퍼런스의 이름 및 약자 매핑 (남녀 공통 적용)

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| **ConfAbbrev** | 컨퍼런스의 약자 명칭 | `a_sun` |
| **Description** | 전체 컨퍼런스 명칭 | `Atlantic Sun Conference` |

### MTeamConferences.csv / WTeamConferences.csv
시즌-팀-컨퍼런스 매핑 파일. 팀이 시즌별로 소속한 컨퍼런스 정보를 확인합니다.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example - MTeamConferences) |
| :--- | :--- | :--- |
| **Season** | 측정 연도 | `1985` |
| **TeamID** | 팀 이름 ID 식별 | `1102` |
| **ConfAbbrev** | 소속된 컨퍼런스 명칭 | `wac` |

### MConferenceTourneyGames.csv / WConferenceTourneyGames.csv
포스트 시즌 1부 컨퍼런스 토너먼트의 각 게임 데이터 결과. (정규 시즌 내 처리분)

| 변수명 (Variable) | 설명 (Description) | 예시 (Example - MConferenceTourney) |
| :--- | :--- | :--- |
| **ConfAbbrev** | 해당 토너먼트를 개최한 컨퍼런스 | `ovc` |
| **Season, DayNum, WTeamID, LTeamID** | 게임 식별을 위한 컬럼 | `2001`, `120`, `1122`, `1369` |

### MSecondaryTourneyTeams.csv / WSecondaryTourneyTeams.csv
NCAA 토너먼트 외 타 포스트 시즌 경기 출전 팀 리스트 (예: NIT).

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| **Season** | 출전 시즌 연도 | `1985` |
| **SecondaryTourney** | 출전 토너먼트의 명칭 | `NIT` |
| **TeamID** | 참가 팀 ID | `1108` |

### MSecondaryTourneyCompactResults.csv / WSecondaryTourneyCompactResults.csv
타 포스트 시즌(NIT, CBI, WNIT 등)의 실제 결승 혹은 게임 데이터 결과 요약.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| **Season..NumOT** | 8개 열. 콤팩트 경기 규정과 동일 | `1985, 136, 1151...` |
| **SecondaryTourney** | 경기가 치러진 2부 리그 명칭 | `NIT` |

### MTeamSpellings.csv / WTeamSpellings.csv
외부 입력과 자동 연동을 위해 사용되는 팀 명칭 오탈자 혹은 다르게 부르는 철자 내역.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| **TeamNameSpelling** | 모두 소문자 형식인 텍스트 철자명 | `a&m-corpus chris` |
| **TeamID** | 실제 팀 식별 번호 | `1394` |

### MNCAATourneySlots.csv / WNCAATourneySlots.csv
NCAA 토너먼트 경기에서 씨드가 어떻게 매칭되고 연결되는지에 대한 정보를 다룸.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example - MNCAATourneySlots) |
| :--- | :--- | :--- |
| **Season** | 적용되는 시즌 연도 | `1985` |
| **Slot** | 포지션, 플레이인 게임 등을 가리키는 게임 구역 및 팀 대진 식별자 | `R1W1` |
| **StrongSeed** | 더 강한 승자가 될 거라 예측되는 시드의 이름 | `W01` |
| **WeakSeed** | 약할 것이라 예측되는 대상 시드의 이름 | `W16` |

### MNCAATourneySeedRoundSlots.csv
각 대결이 열리는 슬롯에서 각 시드가 몇 라운드에서, 보통 며칠에 경기가 진행되는지 구조를 표현합니다. 남성 전용.

| 변수명 (Variable) | 설명 (Description) | 예시 (Example) |
| :--- | :--- | :--- |
| **Seed** | 팀의 토너먼트 초기 시드 | `W01` |
| **GameRound** | 발생한 게임 라운드 횟수 (1/2가 첫주, 결승이 6 등) | `1` |
| **GameSlot** | 해당 라운드에서 진행되는 슬롯 | `R1W1` |
| **EarlyDayNum, LateDayNum** | 게임을 치르는 날짜 범위 | `136`, `137` |
