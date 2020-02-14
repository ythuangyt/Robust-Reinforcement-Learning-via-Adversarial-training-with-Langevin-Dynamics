import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class SimpleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_position = 3):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -10
        self.max_position = 10

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_position, high=self.max_position, shape=(1,), dtype=np.float32)
        self.goal_position = goal_position
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_function(self, x):
    	return np.exp(-(x+4)**2)/1.5 + np.exp(-(x-self.goal_position)**2) + np.sin((x - np.pi)*1.5)/10 + 1

    def step(self, action):

        position = self.state[0]
        position += action[0]
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position

        reward = self.reward_function(position)

        done = bool(position == self.goal_position)
        self.state = np.array([position])
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0])
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
