import torch as th
import os
import gym
from TD3_module import TD3,test
path = os.getcwd()
model_save_path = path + '/results/DDPG/checkpoints/'
env_name='BipedalWalker-v3'
env=gym.make(env_name,render_mode='human')
agent=TD3(env)
agent.actor_net.load_state_dict(th.load(model_save_path+'model_4500_actor.pth'))
agent.critic_net1.load_state_dict(th.load(model_save_path+'model_4500_critic1.pth'))
agent.critic_net2.load_state_dict(th.load(model_save_path+'model_4500_critic2.pth'))
test(env=env,agent=agent,test_episodes=50)