import pandas as pd
# v, ve, vep
# glm_v, glm_e, glm_p
# glm_vep, glm_ve, glm_vp, glm_ep
# glm2_vep, glm2_v, glm2_e, glm2_p
# glm2_v0, glm2_v1, glm2_e0, glm2_e1, glm2_p0, glm2_p1
# glm2_v0e0, glm2_v0p0, glm2_e0p0, glm2_v1e1, glm2_v1p1, glm2_e1p1
# glm2_v0e0p0, glm2_v1e1p1
file_name = 'model_glm2_ep'
file_paths = []

file_num = 12
for i in range(file_num ):
    file_paths.append('./fitting_result/'+file_name+'_'+str(i)+'_exp2.csv')


# 初始化存储不同参数的DataFrame列表
parameters = []
parameter_num = 7
for i in range(parameter_num):
    parameters.append([])



# 读取每个文件并按顺序分割数据
data_num = int(60/file_num)
n_file = -1
for file in file_paths:
    n_file += 1
    if n_file != file_num-1:
        df = pd.read_csv(file)
        for i in range(parameter_num):
            parameters[i].append(df.iloc[data_num*i:data_num*(i+1)])
    else:
        df = pd.read_csv(file)
        for i in range(parameter_num):
            parameters[i].append(df.iloc[(data_num+2)*i:(data_num+2)*(i+1)])

# for file in file_paths:
#     df = pd.read_csv(file)
#     beta_a_list.append(df.iloc[0:data_num])
#     phi_a_list.append(df.iloc[data_num:data_num*2])
#     persev_a_list.append(df.iloc[data_num*2:data_num*3])
#     beta_b_list.append(df.iloc[data_num*3:data_num*4])
#     phi_b_list.append(df.iloc[data_num*4:data_num*5])
#     persev_b_list.append(df.iloc[data_num*5:data_num*6])

# 合并每个参数的数据

data_0 = pd.concat(parameters[0], ignore_index=True)
data_1 = pd.concat(parameters[1], ignore_index=True)
data_2 = pd.concat(parameters[2], ignore_index=True)
data_3 = pd.concat(parameters[3], ignore_index=True)
data_4 = pd.concat(parameters[4], ignore_index=True)
data_5 = pd.concat(parameters[5], ignore_index=True)
data_6 = pd.concat(parameters[6], ignore_index=True)
# data_7 = pd.concat(parameters[7], ignore_index=True)
# data_8 = pd.concat(parameters[8], ignore_index=True)


# beta_a_df = pd.concat(beta_a_list, ignore_index=True)
# phi_a_df = pd.concat(phi_a_list, ignore_index=True)
# persev_a_df = pd.concat(persev_a_list, ignore_index=True)
# beta_b_df = pd.concat(beta_b_list, ignore_index=True)
# phi_b_df = pd.concat(phi_b_list, ignore_index=True)
# persev_b_df = pd.concat(persev_b_list, ignore_index=True)

# 按顺序组合成最终的DataFrame
merged_df = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6], ignore_index=True)
# merged_df = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8], ignore_index=True)

# 保存合并后的数据到新的 CSV 文件
merged_df.to_csv('./fitting_result/'+file_name+'_exp2.csv', index=False)

print("数据已按要求合并并保存'")