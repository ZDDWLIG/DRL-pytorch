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
env_name='BipedalWalker-v3'
env=gym.make(env_name,render_mode='human')
agent=DDPG(env)
agent.actor_net.load_state_dict(th.load(model_save_path+'model_4500_actor.pth'))
agent.critic_net.load_state_dict(th.load(model_save_path+'model_4500_critic.pth'))
test(env=env,agent=agent,test_episodes=50)
