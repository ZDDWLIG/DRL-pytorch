import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from stable_baselines3 import DDPG
from DDPG_module import DDPG,train
import time

env_name='Pendulum-v1'
env=gym.make(id=env_name,render_mode='rgb_array')
state_dim=env.observation_space.sample().shape[0]
action_dim=env.action_space.shape[0]
min_action=env.action_space.low[0]
max_action=env.action_space.high[0]
agent=DDPG(action_dim=action_dim,state_dim=state_dim,min_action=min_action,max_action=max_action)
train(env,agent)
