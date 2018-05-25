import tensorflow as tf
import joblib
from matplotlib import pyplot as plt
import numpy as np
import os
import csv

from statsmodels.nonparametric.smoothers_lowess import lowess

log_dir = "./Data/"
games = ["carnival"]
policies = ["lstm2","qmdp2"]
colors = {"lstm":"green","qmdp":"blue","lstm2":"black","qmdp2":"red"}
occs = [20]

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 

for occ in occs:
	for game in games:
		print(game)
		fig = plt.figure(1,figsize=(6,6))
		lines = []
		for policy in policies:
			name = policy+'_'+game+'Ram'+str(occ)
			print(name)
			reader = csv.DictReader(open(log_dir+"CSV/"+name+'.csv'))
			AvgRewards = []
			MinRewards = []
			MaxRewards = []
			AvgDisRewards = []
			for row in reader:
				# print(row)
				AvgRewards.append(row['avg_reward'])
				MinRewards.append(row['min_reward'])
				MaxRewards.append(row['max_reward'])
				AvgDisRewards.append(row['avg_dis_reward'])
			x = range(len(AvgRewards))

			result = lowess(AvgRewards,x)
			AvgRewards_smooth = result[:,1]

			line, = plt.plot(range(len(AvgRewards_smooth)), AvgRewards_smooth, label=name,color=colors[policy])
			lines.append(line)
		plt.legend(handles=lines)
		plt.xlabel('Iteration')
		plt.ylabel('Average Undiscounted Path Reward')
		# plt.show()
		fig.savefig(log_dir+"Plot/"+game+'Ram'+str(occ)+'.pdf')
		plt.close(fig)



