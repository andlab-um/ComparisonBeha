# ComparisonBeha
The codebase underlying Zhang et al.'s 2023  paper about Irrelevant social comparison affects the exploration of uncertainty and its association with subjective expectations (preprint: https://psyarxiv.com/74qvb/)
![image text](https://github.com/andlab-um/ComparisonBeha/blob/main/Framework_noQ.png)

# Description
Codes for data analysis, behavioral data, and fitting result data in 'Irrelevant social comparison affects the exploration of uncertainty and its association with subjective expectations'.
Behavioral data and fitting result data are stored in csv files and data analysis codes are written in Python. Behavioral data is fitted by the Stan package (pystan).

## Introduction
This project is about 'how irrelevant social comparison influences exploration under uncertainty and its association with subjective expectations'. Our study highlights the critical role of social comparison in the decision-making process and demonstrates how an individual's tendency to make social comparisons shapes their perceptions and behavior strategies.

## Background
The field of exploration behavior has received widespread attention (Wu, 2018, Nat. Hum. Behav., Lee, 2023, Sci. Adv.), but in reality, our behavior strategies are influenced by the social environment and many other factors (Yao, 2022, Nat. Hum. Soc. Sci. Comm, Fan, 2022, Nat. Hum. Behav., Toyokawa, 2019, Nat. Hum. Behav.). For instance, experience can increase the planning depth during decision-making (van Opheusden, 2023, Nat.), people model the minds of others when making decisions within social groups (Khalvati, 2019, Sci. Adv.), and social learning can amplify expressions of anger in social networks (Brady, 2021, Sci. Adv.). However, these studies have not addressed the potential impact of comparison behavior on decision-making when we compare and learn from others, especially when the information provided by these comparisons does not aid our self-assessment, termed as "irrelevant social comparisons."

## Experiment
![image text](https://github.com/andlab-um/ComparisonBeha/blob/main/E1design.png)
To address this issue, we designed three experiments based on the classic restless bandit experiment. 

In Exp1 -- How irrelevant social comparisons affect participants' choosing process, we had two participants perform the experiment together and allowed them to see each other's reward information at the result stage. However, the reward distributions of the two were completely independent, meaning any comparison about rewards was irrelevant. We found that even such irrelevant social comparisons significantly affect participants’ exploitation strategies.

In Exp2 -- How irrelevant social comparisons affect participants' learning process, we additionally asked participants to predict the mean reward for a bandit after receiving reward and report their confidence rating in this prediction. We found that when participants received higher rewards than others, they reported stronger subjective confidence, implying a decrease in their uncertainty.

In Exp2 -- If irrelevant social comparisons affect participants' exploration strategies by changing their subjective expectations, we altered the subjective expectations of participants and had them complete the experiment individually. We found that changing subjective expectations had a consistent impact on participants' exploitation behavior as irrelevant social comparisons, but the impact on exploration behavior was significantly different.

## HISTORY
21.03.2022 - Initiation date



## REFERENCES


## Notes
If you want to run the code, pay attention to the environment configuration and the file path.
