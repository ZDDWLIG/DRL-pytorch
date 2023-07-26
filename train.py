import torch as th
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import random,choice,sample
from collections import deque,namedtuple
from torch.utils.tensorboard import SummaryWriter
import os
from module import DQN,train

from matplotlib import pyplot as plt
#Hyperparameter
lr=5e-4
episodes=int(2e6)
device=th.device('cuda' if th.cuda.is_available() else 'cpu')
gamma=0.99
batch_size=64
#Initial
path = os.getcwd()
model_save_path = path + '/results/checkpoints/'
env_name='LunarLander-v2'
env=gym.make(id=env_name,render_mode='rgb_array')
state_dim=len(env.observation_space.sample())
action_dim=env.action_space.n
agent=DQN(action_dim,state_dim)

#Train
train(env,agent,episodes=episodes)
#Test
# agent.eval_net.load_state_dict(th.load(model_save_path+'model_9000.pth'))
# test_env=gym.make(env_name,render_mode='human')
# test(env=test_env,agent=agent)
