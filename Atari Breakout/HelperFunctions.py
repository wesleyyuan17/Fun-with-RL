def clip_reward(reward):
	# clip reward
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1

def learn(mem_replay, main_dqn, eps, target_dqn, batch_size, gamma, opt):
	'''
	Implements action taking, loss calculation, parameter updating
	Args:
		mem_replay: ReplayMemory object for history in training
		main_dqn: DQN object that gets updated at every step
		eps: double for probability of taking a random action
		target_dqn: DQN object that act as target and is updated periodically
		batch_size: int, number of experiences to get from ReplayMemory object
		gamma: float, discount factor for Bellman equation
		opt: Optimizer that performs parameter update
	Returns:
		loss: double for loss value
	'''
	# Draw a minibatch from the replay memory
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch() 

    # The main network estimates which action is best (in the next state s', new_states is passed!) 
    # for every transition in the minibatch
    arg_q_max = main_dqn.act(new_states)

    # The target network estimates the Q-values (in the next state s', new_states is passed!) 
    # for every transition in the minibatch
    q_vals = target_dqn.forward(new_states)
    double_q = q_vals[range(batch_size), arg_q_max]

    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that 
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q * (1-terminal_flags))

    opt.zero_grad() # zeros gradient buffer to set up for next update

    # Gradient descend step to update the parameters of the main network
    loss = torch.nn.functional.smooth_l1_loss(input=double_q, target=target_q, reduction='mean')
    loss.backward() # send updates to update buffer

    opt.step() # take step using updates in update buffer

    return loss

def update_target_dqn(main_dqn, target_dqn):
	'''
	Updates parameters of target DQN
	Args:
		main_dqn: DQN object that is updated at every step and has same number of parameters as target
		target_dqn: DQN object that serves as target in training
	'''
	torch.save(main_dqn, 'NN')
	target_dqn = torch.load('NN')
	target_dqn.eval()

def EESchedule(frame_number, replay_memory_start_size, eps_initial, eps_final):
	'''
	Determine probability of random action
	Args:
		frame_number: int, which frame is this
		replay_memory_start_size: int, before which always take random action
		eps_initial: float, initial probability of taking random action (1)
		eps_final: float, minimum probabiliyt of taking a random action
	Returns:
		float between 0 and 1 that is the probability of taking a random action
	'''
	if frame_number < replay_memory_start_size:
		return eps_initial
	else:
		return np.max(replay_memory_start_size / frame_number, eps_final)


def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    for idx, frame_idx in enumerate(frames_for_gif): 
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), 
                                     preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}', 
                    frames_for_gif, duration=1/30)