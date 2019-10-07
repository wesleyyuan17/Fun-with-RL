import numpy as np
import numpy.random as npr
import random
import matplotlib.pyplot as plt
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from SwingyMonkey import SwingyMonkey

import torch
import torch.nn as nn

# implement neural net using pytorch
class Neural_Network(nn.Module):
  def __init__(self):
    super(Neural_Network, self).__init__()
    self.input_dim = 4
    self.hidden_dim = 16
    self.output_dim = 2

    self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
    self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=None) # initialize simple NN structure

  def forward(self, X):
    X = torch.relu(self.fc1(X)) # pass through first layer
    X = self.fc2(X) # pass through second layer
    return X

class Learner:
  '''Class type to play the game'''

  def __init__(self, alpha, gamma, batch_size, mem_size):
    self.last_state  = None
    self.last_action = None
    self.last_reward = None
    self.alpha = alpha
    self.gamma = gamma
    self.batch_size = batch_size
    self.mem_size = mem_size
    self.tick = 0
    self.transition_hist = torch.ones(size=(self.mem_size, 2, 4))
    self.action_hist = np.ones(self.mem_size, dtype=int)
    self.reward_hist = np.ones(self.mem_size)
    self.pred_NN = Neural_Network()
    self.target_NN = Neural_Network()
    self.optimizer = torch.optim.Adam(self.pred_NN.parameters(), lr=alpha)
    self.criterion = nn.MSELoss()
    self.score = 0
    self.max_score = 0
    self.count = 0
    self.current = 0
    self.loss_val = []

  def reset(self):
    self.last_state  = None
    self.last_action = None
    self.last_reward = None
    self.score = 0

  def action_callback(self, state):

    self.score = state['score'] # update score
    new_state = self.state_parser(state) # parse new state for input
  	
    if self.count < self.batch_size:
      if self.last_reward is not None:
        self.update_history(new_state)
      new_action = (0 if npr.rand() < 0.1 else 1)
    else:
      self.tick += 1 # track iterations
      if self.last_reward is not None:
        self.update_history(new_state)

      prev_states, prev_actions, prev_rewards = self.mem_replay()
      # prev_states = torch.tensor(prev_states)

      self.optimizer.zero_grad()

      for i in range(self.batch_size):
        # pred_Q = self.pred_NN(prev_states[i,0,:])[prev_actions[i]]

        # ns = prev_states[i,1,:]
        # best_action = self.pred_NN(ns).argmax()
        # q_val = self.target_NN(ns)[best_action]
        # target_Q = prev_rewards[i] + self.gamma*q_val

        # self.optimizer.zero_grad() # zero gradient buffer (order s.t. gradient step only based on prediction)

        # loss = self.loss(pred_Q, target_Q)
        loss = self.loss(s=prev_states[i,0,:], 
                         a=prev_actions[i], 
                         r=prev_rewards[i], 
                         ns=prev_states[i,1,:])

        self.loss_val.append(loss)
        # print(loss.item())
        loss.backward() # set up gradients in buffer
        # print(self.pred_NN.fc1.bias.grad)
        # self.optimizer.step() # perform update

      self.optimizer.step()

      self.tick += 1 # track iterations

      # take action based on epsilon-greedy
      if npr.rand() < max(1/self.tick, 0.1):
        new_action = (0 if npr.rand() < 0.5 else 1)
      else:
        # print(self.pred_NN(torch.tensor(new_state)))
        new_action = self.pred_NN(torch.tensor(new_state)).argmax() # new action

    # update second neural network as needed
    if self.tick % 200 == 0:
      torch.save(self.pred_NN, 'NN')
      self.target_NN = torch.load('NN')
      self.target_NN.eval()

    self.last_action = int(new_action) # update state and action
    self.last_state = new_state

    return self.last_action

  def reward_callback(self, reward):
    '''This gets called so you can see what reward you get.'''

    self.last_reward = reward

  def state_parser(self, state):
    s = [state['tree']['dist'],
        # state['tree']['top'],
        state['tree']['bot'],
        state['monkey']['vel'],
        # state['monkey']['top'],
        state['monkey']['bot']]
    return s

  def update_history(self, new_state):
    self.transition_hist[self.current, ...] = torch.tensor([self.last_state, new_state])
    self.action_hist[self.current] = self.last_action
    self.reward_hist[self.current] = self.last_reward

    self.count = max(self.count, self.current+1)
    self.current = (self.current + 1) % self.mem_size

  def mem_replay(self):
    idx = []
    for _ in range(self.batch_size):
      idx.append(npr.randint(0, self.count))
    return self.transition_hist[idx, :], self.action_hist[idx], self.reward_hist[idx]

  # def loss(self, pred_Q, target_Q):
  #   return (target_Q - pred_Q)**2

  def loss(self, s, a, r, ns):
    pred_Q = self.pred_NN(s)[a]

    best_action = self.pred_NN(ns).argmax()
    q_val = self.target_NN(ns)[best_action]
    target_Q = r + self.gamma*q_val

    return (target_Q - pred_Q)**2

iters = 200
learner = Learner(alpha = 0.1, gamma = 0.7, batch_size=10, mem_size=10000)
scores = []

for ii in range(iters):

  # Make a new monkey object.
  swing = SwingyMonkey(sound=False,            # Don't play sounds.
                       text="Epoch %d" % (ii), # Display the epoch on screen.
                       tick_length=1,          # Make game ticks super fast.
                       action_callback=learner.action_callback,
                       reward_callback=learner.reward_callback)

  # Loop until you hit something.
  while swing.game_loop():
    pass

  # Reset the state of the learner.
  scores.append(learner.score)
  learner.reset()

fig = plt.figure(figsize=(18,6))
plt.plot(np.arange(0,len(scores)), scores)
plt.xlabel('Iteration')
plt.ylabel('Score')

fig = plt.figure(figsize=(18,6))
plt.plot(np.arange(0,len(learner.loss_val)), learner.loss_val)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()