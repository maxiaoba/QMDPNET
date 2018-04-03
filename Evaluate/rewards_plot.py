import tensorflow as tf
import joblib
from matplotlib import pyplot as plt
import numpy as np
import os
import csv

from statsmodels.nonparametric.smoothers_lowess import lowess

log_dir = "../Test_gen/Data/CSV/"
games = ["carnival","space_invaders","star_gunner"]
policies = ["qmdpk3","qmdpk3_2","gru"]
colors = {"gru":"green","qmdp":"blue","qmdpk3":"red","qmdpk3_2":"orange"}
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
			name = policy+'_'+game+'_'+str(occ)
			print(name)
			reader = csv.DictReader(open(log_dir+name+'.csv'))
			AvgRewards = []
			MinRewards = []
			MaxRewards = []
			AvgDisRewards = []
			for row in reader:
				# print(row)
				AvgRewards.append(row['AverageReturn'])
				MinRewards.append(row['MinReturn'])
				MaxRewards.append(row['MaxReturn'])
				AvgDisRewards.append(row['AverageDiscountedReturn'])
			x = range(len(AvgRewards))

			result = lowess(AvgRewards,x)
			AvgRewards_smooth = result[:,1]

			line, = plt.plot(x, AvgRewards_smooth, label=name,color=colors[policy])
			lines.append(line)
		plt.legend(handles=lines)
		plt.xlabel('Iteration')
		plt.ylabel('Average Undiscounted Path Reward')
		# plt.show()
		fig.savefig(log_dir+game+'_'+str(occ)+'.pdf')
		plt.close(fig)



