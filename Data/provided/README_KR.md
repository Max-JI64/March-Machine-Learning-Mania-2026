# [데이터-설명](Data_description.md)

# 데이터셋 설명 (Dataset Description)

매 시즌 디비전 I 대학 농구 팀들 간에 수천 번의 NCAA® 농구 경기가 치러지며, 이는 3월 중순부터 4월 초 챔피언 결정전까지 진행되는 전미 챔피언십 남녀 토너먼트인 '3월의 광란(March Madness®)'으로 정점을 찍습니다. 당사는 오랜 기간에 걸친 대학 농구 경기 및 팀에 대한 방대한 과거 데이터를 제공했습니다. 이 과거 데이터를 활용하여 여러분은 데이터를 탐색하고 March Madness® 경기 결과를 예측하는 자신만의 독특한 방법을 개발할 수 있습니다.

데이터 파일에는 남성 데이터와 여성 데이터가 모두 포함되어 있습니다. 남성 데이터에만 해당하는 파일은 접두사 **M**으로 시작하고, 여성 데이터에만 해당하는 파일은 접두사 **W**로 시작합니다. 도시(Cities) 및 컨퍼런스(Conferences)와 같은 일부 파일은 남성과 여성 데이터를 모두 포함합니다.

아래 하단 섹션에 나열된 **MTeamSpellings** 및 **WTeamSpellings** 파일은 외부 팀 참조를 자체 팀 ID 구조로 매핑하는 데 도움이 될 수 있습니다.
많은 과거 데이터를 제공해주신 Kenneth Massey님께 감사의 말씀을 전합니다.

본 대회를 위한 데이터셋 구성을 지원해주신 Sonas Consulting의 Jeff Sonas님께 특별한 감사를 표합니다.

## 파일 설명 (File descriptions)

아래에서는 대회 데이터 파일의 형식과 필드에 대해 설명합니다. 모든 파일은 이번 시즌 2월 4일까지의 데이터를 포함하고 있습니다. 3월 중순 토너먼트가 가까워짐에 따라, 이번 시즌의 남은 주간 데이터를 반영하여 이 파일들을 업데이트할 예정입니다.

## 데이터 섹션 1 - 기본 사항 (The Basics)

이 섹션은 간단한 예측 모델을 구축하고 예측을 제출하는 데 필요한 모든 것을 제공합니다.

- 팀 ID 및 팀 이름
- 1984-85 시즌 이후의 토너먼트 시드
- 1984-85 시즌 이후의 모든 정규 시즌, 컨퍼런스 토너먼트 및 NCAA® 토너먼트 경기의 최종 점수
- 날짜 및 지역 이름을 포함한 시즌 레벨 세부 정보
- 제출 파일 예시

관례에 따라 특정 시즌을 식별할 때, 시즌이 시작되는 해가 아닌 시즌이 끝나는 해를 기준으로 참조합니다.

### 데이터 섹션 1 파일: MTeams.csv 및 WTeams.csv

이 파일들은 데이터셋에 존재하는 각 대학 팀을 식별합니다.

- **TeamID** - 각 NCAA® 남성 또는 여성 팀을 고유하게 식별하는 4자리 ID 번호입니다. 학교의 TeamID는 해가 바뀌어도 변하지 않으므로, 예를 들어 듀크(Duke) 남성 팀의 TeamID는 모든 시즌에 대해 1181입니다. 남성 팀 ID 범위는 1000-1999이며, 모든 여성 팀 ID 범위는 3000-3999입니다.
- **TeamName** - 팀의 대학 이름을 16자 이하로 요약한 철자입니다.
- **FirstD1Season** - 해당 학교가 디비전 I 학교로 데이터셋에 처음 등장한 시즌입니다. 이 열은 남성 데이터에만 존재하므로 WTeams.csv에는 없습니다.
- **LastD1Season** - 해당 학교가 디비전 I 학교로 데이터셋에 마지막으로 등장한 시즌입니다. 현재 디비전 I인 팀들의 경우 LastD1Season=2026으로 표시됩니다. 이 열 역시 남성 데이터에만 존재합니다.

### 데이터 섹션 1 파일: MSeasons.csv 및 WSeasons.csv

이 파일들은 과거 데이터에 포함된 각 시즌과 해당 시즌 레벨의 속성을 식별합니다.

- **Season** - 토너먼트가 치러진 연도를 나타냅니다.
- **DayZero** - 해당 시즌에서 DayNum=0에 해당하는 날짜입니다. 모든 경기 날짜는 공통 척도에 맞춰 조정되어, (매년) 남성 토너먼트의 월요일 챔피언 결정전이 DayNum=154가 됩니다. 역순으로 거슬러 올라가면, 남성 전미 준결승은 항상 DayNum=152, 남성 "플레이인(play-in)" 경기는 134-135일, 셀렉션 선데이(Selection Sunday)는 132일, 정규 시즌 마지막 날도 132일 등이 됩니다. 날짜 계산을 용이하게 하기 위해 모든 경기 데이터에는 일수(day number)가 포함됩니다. 경기가 치러진 정확한 날짜를 알고 싶다면, 경기의 "DayNum"과 시즌의 "DayZero"를 결합하면 됩니다.
- **RegionW, RegionX, Region Y, Region Z** - 대회 관례에 따라 최종 토너먼트의 4개 지역 각각에 W, X, Y, Z 문자가 할당됩니다. 알파벳 순으로 가장 먼저 오는 지역 이름이 지역 W가 됩니다. 그리고 전미 준결승에서 지역 W와 경기하는 지역이 지역 X가 됩니다. 나머지 두 지역 중 알파벳 순으로 먼저 오는 지역이 지역 Y, 다른 하나가 지역 Z가 됩니다.

### 데이터 섹션 1 파일: MNCAATourneySeeds.csv 및 WNCAATourneySeeds.csv

이 파일들은 모든 과거 데이터 시즌에 대해 각 NCAA® 토너먼트 팀의 시드를 식별합니다.

- **Season** - 토너먼트가 치러진 연도입니다.
- **Seed** - 시드를 식별하는 3자 또는 4자 식별자입니다. 첫 번째 문자는 지역(W, X, Y, Z)을 나타내고, 다음 두 자릿수(01, 02, ..., 16)는 지역 내 시드를 나타냅니다. 플레이인 팀의 경우, 시드를 더 구체적으로 구분하기 위해 네 번째 문자(a 또는 b)가 추가됩니다.
- **TeamID** - MTeams.csv 또는 WTeams.csv 파일에 명시된 팀 ID 번호입니다.

### 데이터 섹션 1 파일: MRegularSeasonCompactResults.csv 및 WRegularSeasonCompactResults.csv

이 파일들은 정규 시즌 경기 결과를 제공합니다. 남성은 1985 시즌부터, 여성은 1998 시즌부터 시작됩니다. "정규 시즌" 경기는 DayNum=132(셀렉션 선데이) 또는 그 이전에 치러진 모든 경기로 정의됩니다.

- **Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT** - 경기 시즌, 일수, 승리 팀 ID, 승리 팀 점수, 패배 팀 ID, 패배 팀 점수, 승리 팀 위치(H: 홈, A: 어웨이, N: 중립), 연장전 횟수를 나타냅니다.

### 데이터 섹션 1 파일: MNCAATourneyCompactResults.csv 및 WNCAATourneyCompactResults.csv

이 파일들은 모든 과거 데이터 시즌의 NCAA® 토너먼트 경기 결과를 식별합니다. 정규 시즌 요약 결과 데이터와 형식이 동일합니다.

### 데이터 섹션 1 파일: SampleSubmissionStage1.csv 및 SampleSubmissionStage2.csv

제출 파일 형식을 보여주는 예시 파일입니다. 모든 가능한 대진에 대해 50%의 승률을 예측한 가장 단순한 제출 형태를 반영합니다.

- **ID** - SSSS_XXXX_YYYY 형식의 14자 문자열입니다 (SSSS: 시즌, XXXX: 낮은 ID 팀, YYYY: 높은 ID 팀).
- **Pred** - ID 필드에 첫 번째로 명시된 팀(XXXX)의 예측 승률입니다.

## 데이터 섹션 2 - 팀 박스 스코어 (Team Box Scores)

이 섹션은 2003 시즌(남성) 또는 2010 시즌(여성) 이후의 모든 경기에 대한 팀 레벨의 경기별 통계(자유투 시도, 수비 리바운드, 실책 등)를 제공합니다.

- **WFGM / LFGM** - 야투 성공 (Field goals made)
- **WFGA / LFGA** - 야투 시도 (Field goals attempted)
- **WFGM3 / LFGM3** - 3점슛 성공
- **WFGA3 / LFGA3** - 3점슛 시도
- **WFTM / LFTM** - 자유투 성공
- **WFTA / LFTA** - 자유투 시도
- **WOR / LOR** - 공격 리바운드
- **WDR / LDR** - 수비 리바운드
- **WAst / LAst** - 어시스트
- **WTO / LTO** - 실책 (Turnovers)
- **WStl / LStl** - 스틸 (Steals)
- **WBlk / LBlk** - 블록 (Blocks)
- **WPF / LPF** - 개인 파울

참고: "야투 성공"(FGM)은 2점슛과 3점슛 성공 횟수를 합친 전체 야투 성공 횟수를 의미합니다.

## 데이터 섹션 3 - 지리 정보 (Geography)

2010 시즌 이후의 모든 경기가 치러진 도시 정보를 제공합니다.

### 데이터 섹션 3 파일: Cities.csv

경기가 개최된 도시의 마스터 목록을 제공합니다. (CityID, City, State)

### 데이터 섹션 3 파일: MGameCities.csv 및 WGameCities.csv

2010 시즌부터의 모든 경기와 해당 경기가 치러진 도시 ID를 매핑합니다.

## 데이터 섹션 4 - 공개 순위 (Public Rankings)

2003 시즌 이후 수십 개의 상위 레이팅 시스템(Pomeroy, Sagarin, RPI, ESPN 등)의 주간 팀 순위(남성 팀 전용)를 제공합니다.

### 데이터 섹션 4 파일: MMasseyOrdinals.csv

다양한 순위 시스템 방법론에 따른 남성 팀의 서수 순위(#1, #2, ...)를 나열합니다.

## 데이터 섹션 5 - 보충 자료 (Supplements)

코치, 컨퍼런스 소속, 대체 팀 이름 철자, 대진표 구조, NIT 등 기타 포스트시즌 토너먼트 경기 결과를 포함한 추가 지원 정보를 포함합니다.

### 주요 파일:
- **MTeamCoaches.csv**: 시즌별 각 팀의 감독 정보
- **Conferences.csv**: 1985년 이후 존재한 디비전 I 컨퍼런스 목록
- **MTeamConferences.csv / WTeamConferences.csv**: 시즌별 팀의 컨퍼런스 소속 정보
- **MTeamSpellings.csv / WTeamSpellings.csv**: 팀 이름의 대체 철자 목록 (외부 데이터 매핑용)
- **MNCAATourneySlots / WNCAATourneySlots**: 토너먼트 대진 구조 및 페어링 매커니즘
