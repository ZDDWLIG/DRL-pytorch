#Use DQN to train and test in gym LunarLander-v2
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
from module import DQN
from matplotlib import pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
#Hyperparameter
lr=5e-4
episodes=2e6
device=th.device('cuda' if th.cuda.is_available() else 'cpu')
gamma=0.99
batch_size=64
#Build DQN
path = os.getcwd()
model_save_path = path + '/results/checkpoints/'


def train(env,agent,episodes=5000,save_inter=500,max_step=1000,epsilon_begin=1.,epsilon_end=0.1,epsilon_decay=0.99,target_score=200):
    if th.cuda.is_available():
        print('Lets use gpu!')
    else:
        print('Lets use cpu!')
    score_list=[]
    eps=epsilon_begin
    flag = False
    writer=SummaryWriter(path+'/results/logs')
    for ep in range(episodes):
        score=0.
        state,_=env.reset()
        for step in range(max_step):
            action=agent.action_sample(state,eps)
            next_state,reward,done,info,_ =env.step(action)
            agent.train(action,state,reward,next_state,done,ep)
            state=next_state
            score+=reward
            if done:
                break
        score_list.append(score)
        score_avg=np.mean(score_list[-100:])
        eps=max(epsilon_end,eps*epsilon_decay)
        writer.add_scalar('score_avg',score_avg,ep)
        if ep%save_inter==0:
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            th.save(agent.eval_net.state_dict(),model_save_path+f'model_{ep}.pth')
            print('episode:{} average_score={}'.format(ep,score_avg))
        if len(score_list)>=100:
            if score_avg>=target_score:
                flag=True
                break
    if flag:
        print('Target achieved!')
    else:
        print('Episodes is too small, target is not achieved...')


def test(env,agent,test_episodes=5,max_step=500):
    score_list=[0]*max_step
    for ep in range(test_episodes):
        state,_=env.reset()
        for step in range(max_step):
            action=agent.action_sample(state,epsilon=0)
            state,reward,done,info,_=env.step(action)
            score_list[step]+=reward
            if done :
                break
    env.close()
    score_list/=test_episodes
    plt.plot([range(max_step),[score_list]])
    plt.xlabel('step')
    plt.ylabel('avg_score')
    plt.show()




#Initial
env_name='LunarLander-v2'
env=gym.make(id=env_name,render_mode='rgb_array')
state_dim=len(env.observation_space.sample())
action_dim=env.action_space.n
agent=DQN(action_dim,state_dim)

#Train
# train(env,agent)
#Test
agent.eval_net.load_state_dict(th.load(model_save_path+'model_1000.pth'))
test_env=gym.make(env_name,render_mode='human')
test(env=test_env,agent=agent)
