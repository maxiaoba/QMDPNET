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
names = ['a2c_lstm16_0','a2c_lstm16_1','ppo2_lstm16_0','ppo2_lstm16_1',\
		'a2c_qmdp_0','a2c_qmdp_1','ppo2_qmdp_0','ppo2_qmdp_1']
labels = names
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
	smooth_strength = 0.99

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

	# result = lowess(AvgRewards,x)
	# AvgRewards_smooth = result[:,1]

	# line, = plt.plot(range(len(AvgRewards_smooth)), AvgRewards_smooth, label=labels[index])#,color=colors[policy])
	line, = plt.plot(range(len(smoothAvgReward)), smoothAvgReward, label=name)#,color=colors[policy])
	lines.append(line)
# plt.title('Deffirent Networks Trained with A2C under Random Seed 0')
plt.legend(handles=lines)
plt.xlabel('Iteration')
plt.ylabel('Average Undiscounted Path Reward')
# plt.show()
fig.savefig(log_dir+"Plot/"+'a2c_ppo2_carnivalRam20'+'.pdf')
plt.close(fig)



