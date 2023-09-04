import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

subject_num = 21
trials = 150

def read_data_task2(n):  #read behavioral data; no action: action[i]=4
    num = str(200+n)
    n_wrong = 0
    #数据文件在的位置
    action = []
    reward = []
    reward_B = []
    comparision = [] #r-r_b; 1:>=0, 0:<0
    expect_arm = []
    expect_reward = []
    expect_confidence = []
    ask = []
       
    file = pd.read_csv('E:/multi-bandit/task2v3/RLE2V3_data/' + num + '.csv')

    action_index = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'choose_bandit.keys':
            action_index = i
    reward_index = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'subchoose':
            reward_index = i
            
    ask_reward_choose_index = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'expectself.text':
            ask_reward_choose_index = i
    ask_reward_random_index = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'expect_random_3.text':
            ask_reward_random_index = i
    ask_reward_confidence = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'slider_2.response':
            ask_reward_confidence = i

    whether_ask = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'whether_ask':
            whether_ask = i        
    ask_random_arm = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'random_whicharm':
            ask_random_arm = i  
        
    for i in range((file.shape[0])):
        if (file.iloc[i,0] == 1 or file.iloc[i,0] == 0) and file.iloc[i,action_index]!='None' :
            # print(i)
            if file.iloc[i,action_index] == 'r':
                action.append(0)
            elif file.iloc[i,action_index] == 'f':
                action.append(1)
            elif file.iloc[i,action_index] == 'i':
                action.append(2)
            elif file.iloc[i,action_index] == 'j':
                action.append(3)
            else:
                print(file.iloc[i,action_index])
                print(i)
                raise ValueError('不能识别选项')
            reward.append(int(file.iloc[i,reward_index]))
            reward_B.append(int(file.iloc[i,reward_index+1]))
            if int(file.iloc[i,reward_index]) >= int(file.iloc[i,reward_index+1]):
                comparision.append(int(file.iloc[i,reward_index])-int(file.iloc[i,reward_index+1]))
            else:
                comparision.append(int(file.iloc[i,reward_index+1])-int(file.iloc[i,reward_index]))
            
            if int(file.iloc[i,whether_ask])==0:
                if np.isnan(file.iloc[i,ask_reward_choose_index]):
                    n_wrong+=1
                    ask.append(2)
                    expect_arm.append(4)
                    expect_reward.append(0)
                    expect_confidence.append(1)
                else: 
                    ask.append(0)
                    expect_arm.append(action[len(action)-1])
                # if np.isnan(file.iloc[i,ask_reward_choose_index]):
                #     print(n,i)
                    expect_reward.append(int(file.iloc[i,ask_reward_choose_index]))
                    expect_confidence.append(float(file.iloc[i,ask_reward_confidence]))
            elif int(file.iloc[i,whether_ask])==1:
                if np.isnan(file.iloc[i,ask_reward_random_index]):
                    n_wrong+=1
                    ask.append(2)
                    expect_arm.append(4)
                    expect_reward.append(0)
                    expect_confidence.append(1)
                else: 
                    ask.append(1)
                    if float(file.iloc[i,ask_random_arm])==-0.6:
                        expect_arm.append(0)
                    elif float(file.iloc[i,ask_random_arm])==-0.2:
                        expect_arm.append(1)
                    elif float(file.iloc[i,ask_random_arm])==0.2:
                        expect_arm.append(2)
                    elif float(file.iloc[i,ask_random_arm])==0.6:
                        expect_arm.append(3)
                    else:
                        print('ERROR expect_arm')
                    expect_reward.append(int(file.iloc[i,ask_reward_random_index]))
                    expect_confidence.append(float(file.iloc[i,ask_reward_confidence]))
            elif int(file.iloc[i,whether_ask])==2 or int(file.iloc[i,whether_ask])==3:
                ask.append(2)
                expect_arm.append(4)
                expect_reward.append(0)
                expect_confidence.append(1)
            else:
                print('ERROR ask whether_whicharm')
                
                
        elif (file.iloc[i,0] == 1 or file.iloc[i,0] == 0)  and file.iloc[i,action_index] =='None' :
            action.append(4)
            reward.append(0)
            reward_B.append(0)
            comparision.append(0)
            expect_arm.append(4)
            expect_reward.append(0)
            expect_confidence.append(1)
            ask.append(2)
    print('subject: ',n,' n_wrong: ',n_wrong)
    return action,reward,reward_B,comparision,expect_arm,expect_reward,expect_confidence,ask

def read_true_reward_distribution(n):
    num = str(200+n)
    file = pd.read_csv('E:/multi-bandit/task2v3/RLE2V3_data/' + num + '.csv')
    reward_0 = []
    reward_1 = []
    reward_2 = []
    reward_3 = []
    
    yellow_index = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'rewardyellow':
            yellow_index = i
    red_index = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'rewardred':
            red_index = i
    blue_index = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'rewardblue':
            blue_index = i
    green_index = 0
    for i in range((file.shape[1])):
        if file.columns[i] == 'rewardgreen':
            green_index = i        
    
    for i in range((file.shape[0])):
        if (file.iloc[i,0] == 1 or file.iloc[i,0] == 0):
            reward_0.append(int(file.iloc[i,yellow_index]))
            reward_1.append(int(file.iloc[i,red_index]))
            reward_2.append(int(file.iloc[i,blue_index]))
            reward_3.append(int(file.iloc[i,green_index]))
    reward_true = np.array([reward_0,reward_1,reward_2,reward_3]).transpose()
    return reward_true

def rewards2comparison(action,reward,reward_B,gamma):
    comparison = []
    com = 0
    for i in range(len(action)):
        if action[i]!=4:
            comparison.append(com)
            com = reward[i] - reward_B[i] + gamma*com
        else:
            comparison.append(0)
            com = gamma*com
    return comparison


action,reward,reward_B,_,expect_arm,expect_reward,expect_confidence,ask =read_data_task2(subject_num)
reward_true = read_true_reward_distribution(subject_num)
comparision = rewards2comparison(action,reward,reward_B,0)
sig_o = 4
sig_d = 2.8
v = np.ones((4))*50
sig = np.ones((4))*4
v_subjective_sub = []
v_normative_sub = []
v_true_sub = []
uncer_subject_sub = []
uncer_normative_sub = []
regression_reward_sub = []
regression_comparison_sub = []
expect_arm_sub = []
for t in range(trials):
    if action[t]!=4:
        pe = reward[t] - v[action[t]]
        Kgain = np.square(sig[action[t]])/(np.square(sig[action[t]])+np.square(sig_o))
        v[action[t]] = v[action[t]] + Kgain * pe
        sig[action[t]] = np.sqrt((1-Kgain)*np.square(sig[action[t]]))
        if ask[t]!=2 :
            v_normative_sub.append(v)
            uncer_normative_sub.append(sig)
            v_true_sub.append(reward_true[t])
            regression_reward_sub.append(reward[t])
            regression_comparison_sub.append(comparision[t])
            
            
            v_subjective_sub.append(expect_reward[t])
            uncer_subject_sub.append(1/expect_confidence[t])
            expect_arm_sub.append(expect_arm[t])
    sig = np.sqrt(np.square(sig)+np.square(sig_d))
    
V_subject = []
V_true = []
comparison = []

for i in range(len(v_subjective_sub)):
    V_subject.append(v_subjective_sub[i])
    V_true.append(v_true_sub[i][expect_arm_sub[i]])
    comparison.append(regression_comparison_sub[i])
data = {'v_subjective':V_subject,'v_true':V_true,'comparison':comparison}
data = pd.DataFrame(data)
lm = ols('v_subjective ~ v_true + comparison',data=data).fit()
print(lm.summary())