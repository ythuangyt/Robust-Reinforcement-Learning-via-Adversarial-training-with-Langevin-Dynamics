# Robust Reinforcement Learning via Adversarial training with Langevin Dynamics

This is the implementations of the paper Robust Reinforcement Learning via Adversarial training with Langevin Dynamics [https://arxiv.org/abs/2002.06063].

## Table of Contents
- [simple_env](#simple_env) 
- [mujoco_env](#mujoco_env) 


## simple_env

## mujoco_env

### Requirements:
* [MuJoCo](http://mujoco.org)
* Python 3 (it might work with Python 2, not tested)
* [PyTorch](http://pytorch.org/)
* [OpenAI Gym](https://github.com/openai/gym)
* [tdqm](https://github.com/tqdm/tqdm)
* numpy
* matplotlib

### How to train
The paper results can be reproduced by running:
```
python run_experiment.py
```
Hyper-parameters, optimizer, and configuration of one-player and two-player can be modified with different arguments to run_experiment.py.

Experiments on single environments can be run by calling:
```
python main.py --env HalfCheetah-v2
```
Hyper-parameters can be modified with different arguments to main.py.

### How to evaluate
Once models has been trained, run:
```
python run_test.py
```
so that you can evaluate several models. Hyper-parameters, optimizer, and configuration of one-player and two-player can be modified with different arguments to run_test.py.

Test on single model can be run by calling:
```
python test.py --env HalfCheetah-v2
```
Hyper-parameters can be modified with different arguments to test.py.

### How to visualize

See `plot.ipynb` for an example of how to access and visualize your models.

