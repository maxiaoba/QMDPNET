from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()
sess.__enter__()
with tf.variable_scope("qmdp_net"):
    obj_qmdp = joblib.load('./Data/Test2/params.pkl')
    rewards_qmdp = obj_qmdp['rewards']
with tf.variable_scope("gru_net"):
    obj_gru = joblib.load('./Data/Test_gru/params.pkl')
    rewards_gru = obj_gru['rewards']


AvgRewards_qmdp = rewards_qmdp['AverageReturn']
MinRewards_qmdp = rewards_qmdp['MinReturn']
MaxRewards_qmdp = rewards_qmdp['MaxReturn']
AvgDisRewards_qmdp = rewards_qmdp['average_discounted_return']  

AvgRewards_gru = rewards_gru['AverageReturn']
MinRewards_gru = rewards_gru['MinReturn']
MaxRewards_gru = rewards_gru['MaxReturn']
AvgDisRewards_gru = rewards_gru['average_discounted_return']

x = range(len(AvgRewards_qmdp))

fig = plt.figure(1)
qmdp_line, = plt.plot(x, AvgRewards_qmdp, label='QMDP-Net')
gru_line, = plt.plot(x, AvgRewards_gru, label='GRU-Net')
plt.legend(handles=[qmdp_line,gru_line])
plt.xlabel('Iteration')
plt.ylabel('Average Undiscounted Path Reward')
# plt.show()
fig.savefig('Reward.pdf')
plt.close(fig)

fig = plt.figure(2)
qmdp_line, = plt.plot(x, AvgDisRewards_qmdp, label='QMDP-Net')
gru_line, = plt.plot(x, AvgDisRewards_gru, label='GRU-Net')
plt.legend(handles=[qmdp_line,gru_line])
plt.xlabel('Iteration')
plt.ylabel('Average Discounted Path Reward')
# plt.show()
fig.savefig('Reward_dis.pdf')
plt.close(fig)


