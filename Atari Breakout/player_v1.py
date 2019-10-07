import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = gym.make('BreakoutDeterministic-v4')
env.reset()

