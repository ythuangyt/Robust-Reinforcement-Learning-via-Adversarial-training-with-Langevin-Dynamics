# Robust Reinforcement Learning via Adversarial training with Langevin Dynamics

This is the implementations of the paper Robust Reinforcement Learning via Adversarial training with Langevin Dynamics [https://arxiv.org/abs/2002.06063]. We demonstrate the effectiveness of using the MixedNE-LD framework to solve the robust RL problems by two kinds of settings, on-policy (VPG) and off-policy experiments.

## Table of Contents
- [simple_env](#simple_env) 
- [mujoco_env](#mujoco_env) 
- [How to train](#how to train) 

## simple_env: On-Policy (VPG) Experiments
Create your own environemt by following create_simple_env folders.

## mujoco_env: Off-Policy (DDPG) Experiments
### Requirements:
* [MuJoCo](http://mujoco.org)
* Python 3 (it might work with Python 2, not tested)
* [PyTorch](http://pytorch.org/)
* [OpenAI Gym](https://github.com/openai/gym)
* [tdqm](https://github.com/tqdm/tqdm)
* numpy
* matplotlib

## How to train
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

If trained with the environment you have created, run
```
python main.py --env simple-v0
```
Pass your own environment name at argument env.

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

