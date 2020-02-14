from gym.envs.registration import register

register(
    id='simple-v0',
    entry_point='gym_simple.envs:SimpleEnv',
)
