# Requirements
- numpy
- pandas
- scipy
- random
- csv
- matplotlib
- seaborn
- statsmodels

# comparison.ipynb
This file was used to generate Figure 4 'Group fitting results of Exp 1 and Exp 3'. (Exp1)

# model_validation.ipynb
This file was used to generate Figure 2D 'the probabilities of exploration in Exp1' and Figure 3 'modeling results of Exp 1' and calculate the regression result of Table 1 'The linear regression between social comparison and belief update of the mean and variance (uncertainty) of the bandits’ reward distributions' and supplementary tables.

# Key functions

## def read_data(n)
This function is used to read real experimental data from participants, including the participants' actions, rewards, Player B's rewards, and the difference in rewards.

## def read_parameter(file_name)
This function is used to read model fitting parameters, including beta_a, phi_a, rho_a, beta_b, phi_b, and rho_b.

## def log_likelyhood(file_name)
This function calculates the log-likelihood of the corresponding model using the model-fitted parameters and participants' real behavioral data.

## def exploration_exploitation()
This function calculates the proportion of exploration by participants when Player B receives more rewards and the proportion of exploration when Player B receives fewer rewards.
