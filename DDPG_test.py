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
from DDPG_module import DDPG,test
from matplotlib import pyplot as plt
path = os.getcwd()
model_save_path = path + '/results/DDPG/checkpoints/'
env_name='Pendulum-v1'
env=gym.make(env_name,render_mode='human')

state_dim=env.observation_space.sample().shape[0]
action_dim=env.action_space.shape[0]
min_action=env.action_space.low[0]
max_action=env.action_space.high[0]
agent=DDPG(action_dim=action_dim,state_dim=state_dim,min_action=min_action,max_action=max_action)
agent.actor_net.load_state_dict(th.load(model_save_path+'model_3500_actor.pth'))
agent.critic_net.load_state_dict(th.load(model_save_path+'model_3500_critic.pth'))
test(env=env,agent=agent,test_episodes=50)
