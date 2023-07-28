import torch as th
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque,namedtuple
from torch.utils.tensorboard import SummaryWriter
import os
from matplotlib import pyplot as plt
device=th.device('cuda' if th.cuda.is_available() else 'cpu')
class ReplayBuffer():
    def __init__(self,mem_size,batch_size):
        self.mem_size=mem_size
        self.batch_size=batch_size
        self.memory=deque(maxlen=self.mem_size)
        self.experience=namedtuple(typename='experience',field_names=['state','action','reward','next_state','done'])
    def add(self,state,action,reward,next_state,done):
        experience=self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    def __len__(self):
        return len(self.memory)
    def sample(self):
        experiences=random.sample(self.memory,k=self.batch_size)
        states=th.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions=th.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards=th.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states=th.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones=th.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return states,actions,rewards,next_states,dones



class Actor(nn.Module):
    def __init__(self,  action_dim,state_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.mlp=nn.Sequential(nn.Linear(state_dim,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,action_dim))
    def forward(self, state):
        x = self.max_action * th.tanh(self.mlp(state))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.mlp=nn.Sequential(nn.Linear(state_dim + action_dim, 512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1))
    def forward(self, state, action):
        x = self.mlp(th.cat([state, action], dim=1))
        return x
class DDPG():
    def __init__(self,action_dim,state_dim,min_action,max_action,batch_size=128,lr=5e-4,gamma=0.99,noise=0.1,mem_size=int(1e6),tau=5e-3,tar_update=5):
        self.action_dim=action_dim
        self.state_dim=state_dim
        self.min_action=min_action
        self.max_action=max_action
        self.batch_size=batch_size
        self.lr=lr
        self.gamma=gamma
        self.noise=noise
        self.mem_size=mem_size
        self.tau=tau
        self.tar_update=tar_update
        self.max_action=max_action
        self.actor_net=Actor(self.action_dim,self.state_dim,self.max_action).to(device)
        self.actor_tar_net=Actor(self.action_dim,self.state_dim,max_action).to(device)
        self.critic_net=Critic(self.action_dim,self.state_dim).to(device)
        self.critic_tar_net=Critic(self.action_dim,self.state_dim).to(device)
        self.actor_tar_net.load_state_dict(self.actor_net.state_dict())
        self.critic_tar_net.load_state_dict(self.critic_net.state_dict())
        self.replay_buffer=ReplayBuffer(self.mem_size,self.batch_size)
        self.loss=nn.MSELoss()
        self.actor_net_opt=optim.Adam(self.actor_net.parameters(),1e-4)
        self.critic_net_opt=optim.Adam(self.critic_net.parameters(),1e-3)
        self.memory=ReplayBuffer(self.mem_size,self.batch_size)
    def tar_net_update(self):
        for eval_param, target_param in zip(self.actor_net.parameters(), self.actor_tar_net.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)
        for eval_param, target_param in zip(self.critic_net.parameters(), self.critic_tar_net.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

    def action_sample(self,state):
        state=th.from_numpy(state).float().to(device)
        action=self.actor_net(state).cpu().data.numpy().flatten()
        action=action+np.random.normal(0,self.noise,size=self.action_dim).clip([self.min_action],[self.max_action])
        return action

    def learn(self,experiences,eps):
        states, actions, rewards, next_states, dones=experiences
        #updata actor network with DPG
        actor_l=-self.critic_net(states,self.actor_net(states)).mean()
        self.actor_net_opt.zero_grad()
        actor_l.backward()
        self.actor_net_opt.step()
        #updata critic netwark with TD
        next_actions=self.actor_tar_net(next_states)
        tar_q=self.critic_tar_net(next_states,next_actions)
        y_t=rewards+self.gamma*(1-dones)*tar_q.detach()#target net dont need to backward, we have "tar_net_updata()"
        eval_q=self.critic_net(states,actions)
        critic_l=self.loss(eval_q,y_t)
        self.critic_net_opt.zero_grad()
        critic_l.backward()
        self.critic_net_opt.step()

        #updata target network
        if eps%self.tar_update==0:
            self.tar_net_update()
    def train(self,state,action,reward,next_state,done,eps):
        self.memory.add(state,action,reward,next_state,done)
        if len(self.memory)>self.memory.batch_size:
            experiences=self.memory.sample()
            self.learn(experiences,eps)


def train(env,agent,episodes=50000,save_inter=500,max_step=200,target_score=250,path=os.getcwd()):
    if th.cuda.is_available():
        print('Lets use gpu!')
    else:
        print('Lets use cpu!')
    score_list=[]
    flag=False
    model_path=path+'/results/DDPG/checkpoints/'
    log_path=path+'/results/DDPG/logs/'
    log_writer = SummaryWriter(log_path)
    for ep in range(episodes):
        score=0.

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        state,_ =env.reset()
        for step in range(max_step):
            action=agent.action_sample(state)
            next_state,reward,done,info,_=env.step(action)
            agent.train(state,action,reward,next_state,done,ep)
            state=next_state
            score+=reward
        score_list.append(score)
        score_avg=np.mean(score_list[-100:])
        log_writer.add_scalar('score_avg',score,ep)
        if ep%save_inter==0:
            th.save(agent.actor_net.state_dict(),model_path+f'model_{ep}_actor.pth')
            th.save(agent.critic_net.state_dict(),model_path+f'model_{ep}_critic.pth')
            print('episode={},average_score={}'.format(ep,score_avg))
        if len(score_list)>-130:
            if score>target_score:
                flag=True
                break
    if flag:
        print('Target achieved!')
    else:
        print('Episodes is too small, target is not achieved...')



def test(env,agent,test_episodes=5,max_step=200):
    score_list=[]
    for ep in range(test_episodes):
        state,_=env.reset()
        score=0.
        for step in range(max_step):
            action=agent.action_sample(state)
            next_state,reward,done,info,_=env.step(action)
            state=next_state
            score+=reward
            if done:
                break
        score_list.append(score)
        score_avg=np.mean(score_list[-100:])
        print(score_avg)
    # plt.plot([x for x in range(max_step)],score_list)
    # plt.xlabel('step')
    # plt.ylabel('average_score')
    # plt.show()




