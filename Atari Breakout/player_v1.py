import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = gym.make('BreakoutDeterministic-v4')
env.reset()

# Clip reward
def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1

# Exploration vs Exploitation Scheduler?

# Run training