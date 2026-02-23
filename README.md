# [March-Machine-Learning-Mania-2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
## [provided-data](Data/provided/README.md) [제공된-데이터](Data/provided/README.md)

# Overview

You will be forecasting the outcomes of both the men's and women's 2026 collegiate basketball tournaments, by submitting predictions for every possible tournament matchup.  
여러분은 모든 가능한 토너먼트 대진에 대한 예측을 제출함으로써, 2026년 남녀 대학 농구 토너먼트의 결과를 예측하게 됩니다.

## Description

Another year, another chance to predict the upsets, call the probabilities, and put your bracketology skills to the leaderboard test. In our twelfth annual March Machine Learning Mania competition, Kagglers will once again join the millions of fans who attempt to predict the outcomes of this year's college basketball tournaments. Unlike most fans, you will pick the winners and losers using a combination of rich historical data and computing power, while the ground truth unfolds on television.  
또 다른 한 해, 이변을 예측하고 확률을 제시하며 여러분의 브래킷 분석(bracketology) 기술을 리더보드에서 시험해 볼 수 있는 또 다른 기회가 찾아왔습니다. 제12회 연례 '3월 머신러닝 매니아' 대회에서 캐글러들은 올해 대학 농구 토너먼트 결과를 예측하려는 수백만 명의 팬들과 다시 한번 함께하게 됩니다. 대부분의 팬들과 달리, 여러분은 TV에서 실제 결과가 펼쳐지는 동안 풍부한 역사적 데이터와 컴퓨팅 파워를 조합하여 승자와 패자를 선택하게 됩니다.

You are provided data of historical NCAA games to forecast the outcomes of the Division 1 Men's and Women's basketball tournaments. This competition is the official 2026 edition, with points, medals, prizes, and basketball glory at stake.  
디비전 1 남녀 농구 토너먼트의 결과를 예측할 수 있도록 과거 NCAA 경기 데이터가 제공됩니다. 이 대회는 공식 2026년 에디션으로, 포인트, 메달, 상금, 그리고 농구의 영광이 걸려 있습니다.

We are continuing the format from last year where you are making predictions about every possible matchup in the tournament, evaluated using the Brier score. See the Evaluation Page for full details.  
작년과 동일한 형식을 유지하여 토너먼트의 모든 가능한 대진에 대해 예측하며, Brier 점수를 사용하여 평가됩니다. 자세한 내용은 평가(Evaluation) 페이지를 참조하세요.

Prior to the start of the tournaments, the leaderboard of this competition will reflect scores from 2021-2025 only. Kaggle will periodically fill in the outcomes and rescore once the 2026 games begin.  
토너먼트 시작 전까지, 본 대회의 리더보드는 2021-2025년의 점수만을 반영합니다. 2026년 경기가 시작되면 캐글은 주기적으로 결과를 채워 넣고 점수를 다시 계산할 것입니다.

Good luck and happy forecasting!  
행운을 빌며 즐거운 예측 되시길 바랍니다!

## Evaluation

Submissions are evaluated on the Brier score between the predicted probabilities and the actual game outcomes (this is equivalent to mean squared error in this context).  
제출물은 예측된 확률과 실제 경기 결과 사이의 Brier 점수로 평가됩니다(이는 이 문맥에서 평균 제곱 오차와 동일합니다).

### Submission File
As a reminder, the submission file format also has a revised format from prior iterations:  
참고로, 제출 파일 형식도 이전 버전들과 비교하여 변경되었습니다.

1. We have combined the Men's and Women's tournaments into one single competition. Your submission file should contain predictions for both.    
남녀 토너먼트를 하나의 단일 대회로 통합했습니다. 여러분의 제출 파일에는 두 토너먼트 모두에 대한 예측이 포함되어야 합니다.

2. You will be predicting the hypothetical results for every possible team matchup, not just teams that are selected for the NCAA tournament. This change was enacted to provide a longer time window to submit predictions for the 2026 tournament. Previously, the short time between Selection Sunday and the tournament tipoffs would require participants to quickly turn around updated predictions. By forecasting every possible outcome between every team, you can now submit a valid prediction at any point leading up to the tournaments.  
NCAA 토너먼트에 선정된 팀뿐만 아니라 모든 가능한 팀 대진에 대한 가상의 결과를 예측하게 됩니다. 이 변화는 2026년 토너먼트 예측을 제출할 수 있는 더 긴 시간 간격을 제공하기 위해 도입되었습니다. 이전에는 '셀렉션 선데이'와 토너먼트 시작 사이의 짧은 시간 때문에 참가자들이 업데이트된 예측을 빠르게 준비해야 했습니다. 모든 팀 간의 모든 가능한 결과를 예측함으로써, 이제 토너먼트가 시작되기 전 언제든지 유효한 예측을 제출할 수 있습니다.

3. You may submit as many times as you wish before the tournaments start, but make sure to select the two submissions you want to count towards scoring. Do not rely on automatic selection to pick your submissions.  
토너먼트 시작 전까지 원하는 만큼 여러 번 제출할 수 있지만, 점수에 반영할 두 개의 제출물을 직접 선택해야 합니다. 자동 선택 기능에 의존하지 마세요.

As with prior years, each game has a unique ID created by concatenating the season in which the game was played and the two team's respective TeamIds. For example, "2026_1101_1102" indicates a hypothetical matchup between team 1101 and 1102 in the year 2026. You must predict the probability that the team with the lower TeamId beats the team with the higher TeamId. Note that the men's teams and women's TeamIds do not overlap.  
이전 연도와 마찬가지로, 각 경기는 경기가 치러진 시즌과 두 팀의 각각의 TeamId를 연결하여 만든 고유 ID를 갖습니다. 예를 들어, "2026_1101_1102"는 2026년에 팀 1101과 팀 1102 간의 가상 대진을 나타냅니다. TeamId가 낮은 팀이 더 높은 팀을 이길 확률을 예측해야 합니다. 남성 팀과 여성 팀의 TeamId는 겹치지 않습니다.

The resulting submission format looks like the following, where Pred represents the predicted probability that the first team will win:  
최종 제출 형식은 다음과 같으며, Pred는 첫 번째 팀이 승리할 예측 확률을 나타냅니다.

```
ID,Pred
2026_1101_1102,0.5
2026_1101_1103,0.5
2026_1101_1104,0.5
...
```

Your 2026 submissions will score 0.0 if you have submitted predictions in the right format. The leaderboard of this competition will be only meaningful once the 2026 tournaments begin and Kaggle rescores your predictions!  
올바른 형식으로 예측을 제출했다면 2026년 제출물은 0.0점을 받게 됩니다. 본 대회의 리더보드는 2026년 토너먼트가 시작되고 캐글이 여러분의 예측을 다시 채점한 후에야 의미가 있을 것입니다!