import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import HelperClasses # contains helpful classes for learning
import HelperFunctions # contains helpful functions

# control parameter
ENV_NAME = 'BreakoutDeterministic-v4' # game environment for gym

MAX_FRAMES = 30000000 # Total number of frames the agent sees
MAX_EPISODE_LENGTH = 18000 # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 200000 # Number of frames the agent sees between evaluations
EVAL_STEPS = 10000 # Number of frames for one evaluation

GAMMA = 0.99 # discount factor for Bellman equation
ALPHA = 0.00001 # learning rate for parameter updates
EPS_INIT = 1 # probability of random action at start
EPS_FINAL = 0.01 # minimum probability of random action

MEMORY_SIZE = 1000000 # max frames stored for memory replay
BATCH_SIZE = 32 # batch size of memory replay
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions, 
                                 # before the agent starts learning
NO_OP_STEPS = 10 # Number of 'NOOP' or 'FIRE' actions at the beginning of an 
                 # evaluation episode
UPDATE_FREQ = 4 # Every four actions a gradient descend step is performed
NETW_UPDATE_FREQ = 10000 # Every C steps, update target network

PATH = "output/" # Gifs and checkpoints will be saved here

# Initialize objects needed
atari = Atari(ENV_NAME)
main_dqn = DQN(atari.env.action_space.n)
target_dqn = DQN(atari.env.action_space.n)
opt = torch.optim.Adam(main_dqn.parameters(), lr=ALPHA)
mem_replay = ReplayMemory()

# Run training
frame_number = 0 # intialize frame count
rewards = [] # rewards experienced
loss_list = [] # losses experienced
while frame_number < MAX_FRAMES:
	''' Training '''
	epoch_frame = 0 # initialize epoch frame count
	while epoch_frame < EVAL_FREQUENCY:
		terminal_life_lost = atari.reset() # initialize state
		episode_reward_sum = 0 # initialize reward count
		for _ in range(MAX_EPISODE_LENGTH):
			# get probability of taking random action and get action
			eps = EESchedule(frame_number, REPLAY_MEMORY_START_SIZE, EPS_INIT, EPS_FINAL)
			action = main_dqn.act(state=atari.state, eps=eps)

			# take step
			processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(action)

			# update counts
			frame_number += 1
			epoch_frame += 1
			episode_reward_sum += reward

			# Clip the reward
			clipped_reward = clip_reward(reward)

			# add experience to memory
			mem_replay.add_experience(action=action, 
									  frame=processed_new_frame[:, :, 0], 
									  reward=clipped_reward, 
									  terminal=terminal_life_lost)

			# calculate loss and update parameters
			if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
				loss = learn(mem_replay, main_dqn, eps, target_dqn, BATCH_SIZE, GAMMA, opt)
				loss_list.append(loss)

			# update target network as necessary
			if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
				update_target_dqn(main_dqn, target_dqn)

			# lost, break out of episode
			if terminal:
				terminal = False
				break

			# record rewards
			rewards.append(episode_reward_sum)

	''' Evaluation '''
	terminal = True
	gif = True
	frames_for_gif = []
	eval_rewards = []
	evaluate_frame_number = 0

	for _ in range(EVAL_STEPS):
		if terminal:
			terminal_life_lost = atari.reset(sess, evaluation=True)
			episode_reward_sum = 0
			terminal = False

	# Fire (action 1), when a life was lost or the game just started, 
	# so that the agent does not stand around doing nothing. When playing 
	# with other environments, you might want to change this...
	action = 1 if terminal_life_lost else main_dqn.act(state=atari.state, eps=0.0) # eps = 0.0

	processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)

	evaluate_frame_number += 1
	episode_reward_sum += reward

	if gif: 
		frames_for_gif.append(new_frame)
	if terminal:
		eval_rewards.append(episode_reward_sum)
		gif = False # Save only the first game of the evaluation as a gif

	print("Evaluation score:\n", np.mean(eval_rewards))       
	try:
		generate_gif(frame_number, frames_for_gif, eval_rewards[0], PATH)
	except IndexError:
		print("No evaluation game finished")

	# Save the network parameters
	torch.save(main_dqn, PATH+'/my_model')
	frames_for_gif = []
