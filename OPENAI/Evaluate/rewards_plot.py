import tensorflow as tf
import joblib
from matplotlib import pyplot as plt
import numpy as np
import os
import csv

from statsmodels.nonparametric.smoothers_lowess import lowess

log_dir = "./Data/"
games = ["carnival"]

# names = []
# for policy in os.listdir(log_dir+"/CSV"):
# 	if policy != '.DS_Store':
# 		names.append(policy[:-4:])
# print(names)
# names = ['a2c_lstm_0','a2c_lstm_1','a2c_lstm32_0','a2c_lstm16_0','a2c_lstm16_1']
# names = ['a2c_lstm16_1','a2c_pifc2_1','a2c_pifc_1',\
# 			'a2c_shallow_1','a2c_dc_1',]
names = ['ppo2_lstm16_1','ppo2_pifc2_1','ppo2_pifc_1',\
			'ppo2_shallow_1','ppo2_dc_1',]
# names = ['a2c_lstm16_lr0.01_0','a2c_pifc_lr0.01_0','a2c_pifc2_lr0.01_0',\
# 			'a2c_shallow_lr0.01_0','a2c_k1_lr0.01_0','a2c_dc_lr0.01_0']
# labels = names
# names = ['a2c_lstm16_0','a2c_lstm16_1','a2c_pifc2_0','a2c_pifc2_1']
# names = ['ppo2_lstm16_0','ppo2_lstm16_1','ppo2_pifc2_0','ppo2_pifc2_1']
labels = ['lstm','qmdp','qmdp_relu',\
			'qmdp_shallow','qmdp_dc']
# colors = {"lstm2":"green","qmdp_pifc":"blue","qmdp_pifc2":"red","qmdp4":"black"}
occs = [20]

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 


fig = plt.figure(1,figsize=(8,8))
lines = []

for (index,name) in enumerate(names):
	print(name)
	reader = csv.DictReader(open(log_dir+"CSV/"+name+'.csv'))
	AvgRewards = []
	MinRewards = []
	MaxRewards = []
	AvgDisRewards = []

	smoothAvgReward = []
	smooth_strength = 0.999

	for (i,row) in enumerate(reader):
		# print(row)
		AvgRewards.append(float(row['avg_reward']))
		MinRewards.append(float(row['min_reward']))
		MaxRewards.append(float(row['max_reward']))

		if i == 0:
			smoothAvgReward.append(AvgRewards[i])
		else:
			smoothAvgReward.append((1-smooth_strength)*AvgRewards[i]+smooth_strength*smoothAvgReward[i-1])
		# AvgDisRewards.append(row['avg_dis_reward'])
	x = range(len(AvgRewards))

	result = lowess(AvgRewards,x)
	AvgRewards_smooth = result[:,1]

	line, = plt.plot(range(len(AvgRewards_smooth)), AvgRewards_smooth, label=labels[index])#,color=colors[policy])
	# line, = plt.plot(range(len(smoothAvgReward)), smoothAvgReward, label=name)#,color=colors[policy])
	lines.append(line)
# plt.title('Deffirent Networks Trained with A2C under Random Seed 0')
plt.legend(handles=lines)
plt.xlabel('Iteration')
plt.ylabel('Average Undiscounted Path Reward')
# plt.show()
fig.savefig(log_dir+"Plot/Report/"+'ppo2_carnivalRam20_seed1'+'.pdf')
plt.close(fig)



