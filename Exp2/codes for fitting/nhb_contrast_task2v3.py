import numpy as np
import pandas as pd
import csv
import pystan
import scipy

def read_data(n):
    #数据文件在的位置
    action = []
    reward = []
    reward_B = []
    comparision = [] #r-r_b; 1:>=0, 0:<0
    # num = str(220+n)
    num = str(220+n)
    file = pd.read_csv('./RLE2V3_data/'+num+'.csv')
    action_index = 0
    for i in  range((file.shape[1])):
        if file.columns[i] == 'choose_bandit.keys':
            action_index = i
    reward_index = 0
    for i in  range((file.shape[1])):
        if file.columns[i] == 'subchoose':
            reward_index = i
    for i in range((file.shape[0])):
        if (file.iloc[i,0] == 1 or file.iloc[i,0] == 0) and file.iloc[i,action_index]!='None' :
            if file.iloc[i,action_index] == 'r':
                action.append(1)
            elif file.iloc[i,action_index] == 'f':
                action.append(2)
            elif file.iloc[i,action_index] == 'i':
                action.append(3)
            elif file.iloc[i,action_index] == 'j':
                action.append(4)
            else:
                print(file.iloc[i,action_index])
                print(i)
                raise ValueError('不能识别选项')
            if int(file.iloc[i,reward_index])>100:
                reward.append(100)
            elif int(file.iloc[i,reward_index])<0:
                reward.append(0)
            else:
                reward.append(int(file.iloc[i,reward_index]))

            if int(file.iloc[i,reward_index+1])>100:
                reward_B.append(100)
            elif int(file.iloc[i,reward_index+1])<0:
                reward_B.append(0)
            else:
                reward_B.append(int(file.iloc[i,reward_index+1]))
            if int(file.iloc[i,reward_index]) >= int(file.iloc[i,reward_index+1]):
                comparision.append(1)
            else:
                comparision.append(0)
        elif (file.iloc[i,0] == 1 or file.iloc[i,0] == 0) and file.iloc[i,action_index] == 'None' :
            action.append(0)
            reward.append(0)
            reward_B.append(0)
            comparision.append(0)
    return action,reward,reward_B,comparision

bandit_model = """
data {
  int<lower=1> subjects;
  int<lower=1> trials;       
  int<lower=0,upper=4> action[subjects, trials];     
  real<lower=0, upper=100> reward[subjects, trials];
  real<lower=-100, upper=100> comparision[subjects, trials];
  }

transformed data {
  real<lower=0, upper=100> v1;
  real<lower=0> sig1;
  real<lower=0> sigO;
  real<lower=0> sigD;

  
  // random walk parameters 
  v1   = 50.0;        // prior belief mean reward value trial 1
  sig1 = 4.0;         // prior belief variance trial 1
  sigO = 4.0;         // observation variance
  sigD = 2.8;         // diffusion variance

}

parameters {
  real<lower=0,upper=3> beta_1[subjects]; 
  real<lower=0,upper=3> phi_1[subjects];
  real<lower=0,upper=10> persev_1[subjects];

  real<lower=0,upper=3> beta_0[subjects]; 
  real<lower=0,upper=3> phi_0[subjects];
  real<lower=0,upper=10> persev_0[subjects];
}

model {

  for (s in 1:subjects) {


    vector[4] v;   // value (mu)
    vector[4] sig; // sigma
    vector[4] eb;  // exploration bonus
    vector[4] pb;  // perseveration bonus

    vector[4] min_sig;
    vector[4] relative_sig;
    real pe;       // prediction error
    real Kgain;    // Kalman gain
  
    v   = rep_vector(v1, 4);
    sig = rep_vector(sig1, 4);
  


  
    for (t in 1:trials) {        
      
    if (action[s,t] != 0) {
      if (t<2) {
        min_sig = rep_vector(min(sig), 4);
        relative_sig = sig - min_sig;  
        eb = (phi_1[s]) * relative_sig;
        pb = rep_vector(0.0, 4);

          
        if (t>1) {
          if (action[s,t-1] !=0) {
            pb[action[s,t-1]] = persev_1[s];
          } 
        }
        action[s,t] ~ categorical_logit(  (beta_1[s] * v + eb + pb) ); // compute action probabilities
      }

      if (t>1) {
        if (comparision[s,t-1]>=0) {  
          min_sig = rep_vector(min(sig), 4);
          relative_sig = sig - min_sig;  
          eb = (phi_1[s]) * relative_sig;
          pb = rep_vector(0.0, 4);

            
          if (t>1) {
            if (action[s,t-1] !=0) {
              pb[action[s,t-1]] = persev_1[s];
            } 
          }
          action[s,t] ~ categorical_logit(  (beta_1[s] *v + eb + pb) ); // compute action probabilities
        }
        if (comparision[s,t-1]<0) {  
          min_sig = rep_vector(min(sig), 4);
          relative_sig = sig - min_sig;  
          eb = (phi_0[s]) * relative_sig;
          pb = rep_vector(0.0, 4);
            
          if (t>1) {
            if (action[s,t-1] !=0) {
              pb[action[s,t-1]] = persev_0[s];
            } 
          }
          action[s,t] ~ categorical_logit( (beta_0[s] * v + eb + pb) ); // compute action probabilities
        }
      }  
  
      pe    = reward[s,t] - v[action[s,t]];                       // prediction error 
      Kgain = sig[action[s,t]]^2 / (sig[action[s,t]]^2 + sigO^2); // Kalman gain
        
      v[action[s,t]]   = v[action[s,t]] + Kgain * pe;             // value/mu updating (learning)
      sig[action[s,t]] = sqrt( (1-Kgain) * sig[action[s,t]]^2 );  // sigma updating
        
    }
      

    for (j in 1:4) {
        sig[j] = sqrt( sig[j]^2 + sigD^2 );
    }
    }
    
  }  
}
"""

def reward2comparison(action,reward,reward_B,gamma):
    comparison = []
    com = 0
    for i in range(len(reward)):
        if action[i]!=0:
            if com<0:
                comparison.append(-np.sqrt(-com))
            else:
                comparison.append(np.sqrt(com))
            com = reward[i]-reward_B[i] + gamma*com
            com = min(com,100)
            com = max(-100,com)
        else:
            comparison.append(0)
            com = gamma*com
    return comparison


subject = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
action = []
reward = []
reward_B = []
comparision = []
for i in subject:
    a, r, r_b,_ = read_data(i)
    action.append(a)
    reward.append(r)
    compare = reward2comparison(a,r,r_b,0)
    comparision.append(compare)

action = np.array(action)
reward = np.array(reward)
comparision = np.array(comparision)

bandit_data = {"subjects":len(subject),
               "trials":len(action[0]),
               "action":action,
               "reward":reward,
               "comparision":comparision}
sm = pystan.StanModel(model_code=bandit_model)
fit = sm.sampling(data=bandit_data,iter=20000,chains=4,warmup=16000,control={'adapt_delta':0.8 ,'max_treedepth':10})
summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'],columns=summary_dict['summary_colnames'],index=summary_dict['summary_rownames'])
df.to_csv('nhb_contrast.csv')

subject = [31,32,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]
action = []
reward = []
reward_B = []
comparision = []
for i in subject:
    a, r, r_b,_ = read_data(i)
    action.append(a)
    reward.append(r)
    compare = reward2comparison(a,r,r_b,0)
    comparision.append(compare)

action = np.array(action)
reward = np.array(reward)
comparision = np.array(comparision)

bandit_data = {"subjects":len(subject),
               "trials":len(action[0]),
               "action":action,
               "reward":reward,
               "comparision":comparision}
sm = pystan.StanModel(model_code=bandit_model)
fit = sm.sampling(data=bandit_data,iter=20000,chains=4,warmup=16000,control={'adapt_delta':0.8 ,'max_treedepth':10})
summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'],columns=summary_dict['summary_colnames'],index=summary_dict['summary_rownames'])
df.to_csv('nhb_contrast_1.csv')