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
from module import DQN,test
from matplotlib import pyplot as plt
path = os.getcwd()
model_save_path = path + '/results/checkpoints/'
env_name='LunarLander-v2'
test_env=gym.make(env_name,render_mode='rgb_array')

state_dim=len(test_env.observation_space.sample())
action_dim=test_env.action_space.n
agent=DQN(action_dim,state_dim)
agent.eval_net.load_state_dict(th.load(model_save_path+'model_4500.pth'))
test(env=test_env,agent=agent,test_episodes=50)
