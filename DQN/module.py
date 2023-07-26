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
from matplotlib import pyplot as plt
#Hyperparameter
lr=1e-3
episodes=2e6
device=th.device('cuda' if th.cuda.is_available() else 'cpu')
gamma=0.99
batch_size=64
#Build DQN
class Q_Net(nn.Module):
    def __init__(self,action_dim,state_dim):
        super().__init__()
        self.fc1=nn.Linear(state_dim,64)
        self.fc2=nn.Linear(64,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,action_dim)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer():
    def __init__(self,action_dim,mem_size,batch_size):
        self.action_dim=action_dim
        self.mem_size=mem_size
        self.batch_size=batch_size
        self.memory=deque(maxlen=self.mem_size)
        self.experience=namedtuple(typename='experience',field_names=['action','state','reward','next_state','done'])
        #get current length
    def __len__(self):
        return len(self.memory)

    #add experience to memory
    def add(self,action,state,reward,next_state,done):
        experience=self.experience(action,state,reward,next_state,done)
        self.memory.append(experience)

    #get experiences
    def sample(self):
        experiences=sample(self.memory,k=self.batch_size)
        #build batch eg.action->action
        # actions=np.vstack([e.action for e in experiences if e is not None])
        # states=np.vstack([e.state for e in experiences if e is not None])
        # rewards=np.vstack([e.reward for e in experiences if e is not None])
        # next_states=np.vstack([e.next_state for e in experiences if e is not None])
        # dones=np.vstack([e.done for e in experiences if e is not None])
        # actions=th.from_numpy(actions).float().to(device)
        # states=th.from_numpy(states).float().to(device)
        # rewards=th.from_numpy(rewards).float().to(device)
        # next_states=th.from_numpy(next_states).float().to(device)
        # dones=th.from_numpy(dones.astype(np.uint8)).to(device)
        states = th.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = th.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = th.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = th.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = th.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return actions, states, next_states, rewards, dones



class DQN():
    def __init__(self,action_dim,state_dim,batch_size=64,lr=5e-4,gamma=0.99,mem_size=int(1e5),learn_step=5,tau=1e-3,tar_updata=10):
        #parameters
        self.action_dim=action_dim
        self.state_dim=state_dim
        self.batch_size=batch_size
        self.gamma=gamma
        self.learn_step=learn_step
        self.tau=tau
        self.tar_update=tar_updata
        #build model
        self.eval_net=Q_Net(action_dim,state_dim).to(device)
        self.tar_net=Q_Net(action_dim,state_dim).to(device)
        self.opt=optim.Adam(self.eval_net.parameters(),lr)
        self.loss=nn.MSELoss()
        #replay buffer
        self.memory=ReplayBuffer(action_dim,mem_size,batch_size)
        self.counter=0
    #action sample
    def action_sample(self,state,epsilon):
        state=th.from_numpy(state).float().unsqueeze(0).to(device)
        with th.no_grad():
            action_q_value=self.eval_net(state)
        #epsilon greedy
        if random()<epsilon:
            action=choice(np.arange(self.action_dim))
        else:
            action=np.argmax(action_q_value.cpu().data.numpy())
        return action

    #updata target network [W_tar <-- tau*W_eval+(1-tau)*W_tar]
    def tar_net_update(self):
        for eval_param, target_param in zip(self.eval_net.parameters(), self.tar_net.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self,experiences,eps):
        actions, states,next_states , rewards, dones=experiences
        tar_q=self.tar_net(next_states).detach().max(axis=1)[0].unsqueeze(1)#maxQ* from target network
        y_t=rewards+self.gamma*tar_q*(1-dones)#TD target
        eval_q=self.eval_net(states).gather(1,actions.to(int))#Q from eval network

        #backward
        l=self.loss(eval_q,y_t)
        self.opt.zero_grad()
        l.backward()
        self.opt.step()
        #update target net
        if eps %self.tar_update==0:
            self.tar_net_update()


    def train(self,action,state,reward,next_state,done,eps):
        self.memory.add(action,state,reward,next_state,done)
        self.counter+=1
        if self.counter%self.learn_step==0:
            if len(self.memory)>=self.batch_size:
                experiences=self.memory.sample()
                self.learn(experiences,eps)

def train(env,agent,episodes=50000,save_inter=500,max_step=1000,epsilon_begin=1.,epsilon_end=0.1,epsilon_decay=0.99,target_score=200,path=os.getcwd()):
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
        model_save_path = path + '/results/checkpoints/'
        if ep%save_inter==0:
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            th.save(agent.eval_net.state_dict(),model_save_path+f'model_{4500+ep}.pth')
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
    score_list=np.divide(np.array(score_list),test_episodes).tolist()
    plt.plot([x for x in range(max_step)],score_list)
    plt.xlabel('step')
    plt.ylabel('avg_score')
    plt.show()


