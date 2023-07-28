# DRL-pytorch
DQN, PPO, SAC...in pytorch for gym envs
# Installation
Clone this repo:

```
  git clone https://github.com/ZDDWLIG/DRL-pytorch.git
  cd path/to/DRL-pytorch-master
```

Use conda to manage the python environment:

```
  conda create -n DRL-pytorch python=3.8
  conda activate DRL-pytorch
  pip install -r requirements.txt
```
# Train

For example, if you want to train DQN, then run：

```
  cd DQN
  python train.py
```
The checkpoints and logs can be found in `results`

# Test

If you want to test DQN, then run:

```
  cd DQN
  python test.py
```
## Demo
![Lun GIF](https://github.com/ZDDWLIG/DRL-pytorch/blob/master/Lun_over.gif)
