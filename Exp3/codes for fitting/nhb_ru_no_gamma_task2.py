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
    num = str(330+n)
    file = pd.read_csv('./task3_pilot/'+num+'.csv')
    action_index = 0
    for i in  range((file.shape[1])):
        if file.columns[i] == 'choose_bandit.keys':
            action_index = i
    reward_index = 0
    for i in  range((file.shape[1])):
        if file.columns[i] == 'subchoose':
            reward_index = i
    for i in range((file.shape[0])):
        if (file.iloc[i,4] == 1 or file.iloc[i,4] == 0) and file.iloc[i,action_index]!='None' :
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

        elif (file.iloc[i,4] == 1 or file.iloc[i,4] == 0) and file.iloc[i,action_index] == 'None' :
            action.append(0)
            reward.append(0)
            reward_B.append(0)

    return action,reward,reward_B

bandit_model_0 = """
data {
  int<lower=1> subjects;
  int<lower=1> trials;       
  int<lower=0,upper=4> action[subjects, trials];     
  real<lower=0, upper=100> reward[subjects, trials];
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
  real<lower=0,upper=3> beta[subjects]; 
  real<lower=0,upper=10> phi[subjects];
  real<lower=0,upper=10> persev[subjects];
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
      min_sig = rep_vector(min(sig), 4);
      relative_sig = sig - min_sig;
      eb = phi[s] * relative_sig;
      pb = rep_vector(0.0, 4);
        
      if (t>1) {
        if (action[s,t-1] !=0) {
          pb[action[s,t-1]] = persev[s];
        } 
      }
      action[s,t] ~ categorical_logit( (beta[s] * v + eb + pb ) ); // compute action probabilities


      
  
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




subject = np.arange(1,21)
action = []
reward = []
reward_B = []

for i in subject:
    a, r, r_b = read_data(i)
    action.append(a)
    reward.append(r)


action = np.array(action)
reward = np.array(reward)

print(len(action[0]))
bandit_data = {"subjects":len(subject),
               "trials":len(action[0]),
               "action":action,
               "reward":reward}

# posterior = stan.build(bandit_model,data=bandit_data,random_seed=1)
# fit = posterior.sample(num_chains=4, num_samples=5000)
# # eta = fit["eta"]  # array with shape (8, 4000)
# df = fit.to_frame()  # pandas `DataFrame, requires pandasC
# df.to_csv('bayesian_learning.csv')
sm = pystan.StanModel(model_code=bandit_model_0)
fit = sm.sampling(data=bandit_data,iter=20000,chains=4,warmup=16000,control={'adapt_delta':0.8 ,'max_treedepth':10})
summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'],columns=summary_dict['summary_colnames'],index=summary_dict['summary_rownames'])
df.to_csv('nhb_ru_no_gamma_task3_3.csv')