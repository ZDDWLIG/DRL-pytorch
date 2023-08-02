import gym
from DDPG_module import DDPG,train

env_name='Pendulum-v1'
env=gym.make(id=env_name,render_mode='rgb_array')
agent=DDPG(env)
train(env,agent)
