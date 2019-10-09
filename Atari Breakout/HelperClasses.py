## imports for whatever classes need ##############################################################
import numpy as np
from PIL import Image
import torch

## Frame Processing class #########################################################################

class FrameProcessor():
	def __init__(self, frame_height=84, frame_width=84):
		'''
		Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
		'''
		self.frame_height = frame_height
        self.frame_width = frame_width

    def __call__(self, frame):
    	'''
        Args:
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        '''
        processed_frame = Image.fromarray(frame).convert('L') # read in frame, convert to grayscale
        processed_frame = processed_frame.resize((110, 84)) # resize to 110x84 image
        crop_frame = (0, 36, self.frame_width, 36 + self.frame_height) # left, upper, right, lower
        processed_frame = processed_frame.crop(crop_frame) # crop to 84x84 image
        return processed_frame

## Neural Network implementation class ############################################################

class DQN(torch.nn.Module):
	def __init__(self, n_actions, n_filters=[32, 64, 64, 1024], agent_history_length=4):
		'''
		Args:
			n_actions: Integer, number of valid actions for a state i.e. output dim
			n_filters: number of filters in each convolutional layer
			agent_history_length: number of frames that make up state i.e. input dim for first
								  convolutional layer
		'''
        self.action_space = n_actions

		# 4 input layers, 32 filters of 8x8 layer w/ stride 4, apply rectifier
		# output is 20x20x32
		self.conv1 = torch.nn.functional.relu(torch.nn.Conv2d(agent_history_length, 
															  n_filters[0], 
															  kernel_size=(8,8), 
															  stride=4,
															  bias=False))
		# 32 input layers, 64 filters of 4x4 layer w/ stride 2, apply rectifier
		# output is 9x9x64
		self.conv2 = torch.nn.functional.relu(torch.nn.Conv2d(n_filters[0],
															  n_filters[1],
															  kernel_size=(4,4),
															  stride=2,
															  bias=False))
		# 64 input layers, 64 filters of 3x3 layer w/ stride 1, apply rectifier
		# output is 7x7x64
		self.conv3 = torch.nn.functional.relu(torch.nn.Conv2d(n_filters[1],
															  n_filters[2],
															  kernel_size=(3,3),
															  stride=1,
															  bias=False))
		# 64 input layers, 1024 filters of 7x7 layer w/ stride 1, apply rectifier
		# output is 1x1x1024
		self.conv4 = torch.nn.functional.relu(torch.nn.Conv2d(n_filters[2],
															  n_filters[3],
															  kernel_size=(7,7),
															  stride=1,
															  bias=False))
		# advantage stream
		self.action = torch.nn.Sequential(
			torch.nn.Linear(512,512),
			torch.nn.ReLu(),
			torch.nn.Linear(512,n_actions)
			)

		# value stream
		self.value = torch.nn.Sequential(
			torch.nn.Linear(512,512),
			torch.nn.ReLu(),
			torch.nn.Linear(512,1)
			)

	def forward(self, x):
		'''
        Passes x through the DQN, returns the resulting Q-value output
		Args:
			x: A (84, 84, 4) tensor representing current state of game
		Returns:
			Vector of predicted value of taking any of n_action actions given current state
		'''
		x = self.conv1(x) # pass through first convolutional layer
		x = self.conv2(x) # pass through second convolutional layer
		x = self.conv3(x) # pass through third convolutional layer
		x = self.conv4(x) # pass through fourth convolutional layer
		value, advantage = torch.split(x, 2, 3)
		value = self.value(torch.flatten(value)) # value of being in state
		advantage = self.advantage(torch.flatten(advantage)) # vector of advantage for each action
		return value + (advantage - np.mean(advantage))

    def act(self, state, eps):
        '''
        Takes a state and decides how to act for next step
        Args:
            state: A (84, 84, 4) tensor corresponding to last 4 frames seen
            eps: float, epsilon for exploration vs exploitation
        Returns:
            action: integer, index corresponding to some action in action space
        '''
        if np.random.random() < eps: # exploit
            state = torch.tensor(state)
            q_val = self.forward(state)
            action = np.argmax(q_val)
        else: # explore
            action = np.random.randint(0, self.action_space)

        return action

## Exploration v Exploitation scheduler ###########################################################

## Experience Replay class ########################################################################

class ReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, size=1000000, frame_height=84, frame_width=84, 
                 agent_history_length=4, batch_size=32):
        '''
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        '''
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        
        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length, 
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length, 
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        
    def add_experience(self, action, frame, reward, terminal):
        '''
        Args:
            action: An integer between 0 and n_action - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        '''
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
             
    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]
        
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index
            
    def get_minibatch(self):
        '''
        Returns a minibatch of self.batch_size = 32 transitions
        '''
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
        
        self._get_valid_indices()
            
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        
        return [np.transpose(self.states, axes=(0, 2, 3, 1)), 
        		self.actions[self.indices], 
        		self.rewards[self.indices], 
        		np.transpose(self.new_states, axes=(0, 2, 3, 1)), 
        		self.terminal_flags[self.indices]]

## Atari wrapper ##################################################################################

class Atari():
    ''' Wrapper for environment provided by gym '''
    def __init__(self, envName, no_op_steps=10, agent_history_length=4):
        '''
        Args:
            envName: type of environment to pass to gym.make()
            no_op_steps: Integer, random number of fire actions
            agent_history_length: Integer, number of frames that make up a state
        '''
        self.env = gym.make(envName)
        self.process_frame = FrameProcessor()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length

    def reset(self, evaluation=False):
        '''
        Resets environment and stakcs 4 frames on top of each other to produce first state
        '''
        frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True # Set to true so that the agent starts 
                                  # with a 'FIRE' action when evaluating

        if evaluation:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1) # Action 'Fire'
        processed_frame = self.process_frame(frame) # process frame
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2) # stack to create initial state

        return terminal_life_lost

    def step(self, action):
        '''
        Takes action and observes reward and resulting state
        Args:
            action: Integer, corresponds to a valid action to be taken
        '''
        new_frame, reward, terminal, info = self.env.step(action) # take step with gym

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_new_frame = self.process_frame(new_frame) # process frame
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2) # create new state
        self.state = new_state
        
        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame























