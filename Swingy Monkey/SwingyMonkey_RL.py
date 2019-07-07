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
    self.hidden_dim = 4
    self.output_dim = 2

    self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
    self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=None) # initialize simple NN structure

  def forward(self, X):
    X = torch.relu(self.fc1(X)) # pass through first layer
    X = self.fc2(X) # pass through second layer
    return X

class Learner:
  '''Class type to play the game'''

  def __init__(self, alpha, gamma):
    self.last_state  = None
    self.last_action = None
    self.last_reward = None
    self.alpha = alpha
    self.gamma = gamma
    self.tick = 0
    self.transition_hist = []
    self.NN_A = Neural_Network()
    self.NN_B = Neural_Network()
    self.optimizer = torch.optim.Adam(self.NN_A.parameters(), lr=alpha)
    self.criterion = nn.MSELoss()
    self.score = 0
    self.max_score = 0
    self.loss_val = []

  def reset(self):
    self.last_state  = None
    self.last_action = None
    self.last_reward = None
    self.score = 0

  def action_callback(self, state):

    self.score = state['score'] # update score
    new_state = self.state_parser(state) # parse new state for input
  	
    if self.last_reward is None: # take action randomly
      new_action = (0 if npr.rand() < 0.1 else 1)

      self.tick += 1 # track iterations
    else:
      # if npr.rand() < 0.5: # pick one NN to update

      #   update_A = True # track which NN is being updated

      #   pred_Q = self.NN_A(self.last_state)[self.last_action]
      #   a_star = self.NN_A(new_state).argmax()
      #   target = self.last_reward + self.gamma*self.NN_B(new_state)[a_star]
      # else:

      #   update_A = False

      #   pred_Q = self.NN_B(self.last_state)[self.last_action]
      #   a_star = self.NN_B(new_state).argmax()
      #   target = self.last_reward + self.gamma*self.NN_A(new_state)[a_star]
      
      pred_Q = self.NN_A(self.last_state)[self.last_action]
      a_star = self.NN_A(new_state).argmax()
      target = self.last_reward + self.gamma*self.NN_B(new_state)[a_star]

      self.optimizer.zero_grad() # zero gradient buffer

      loss = self.loss(pred_Q, target)
      self.loss_val.append(loss)
      # print(loss.item())
      loss.backward() # set up gradients in buffer
      # print(self.NN_A.fc1.bias.grad)
      self.optimizer.step() # perform update

      self.tick += 1 # track iterations

      # take action based on epsilon-greedy
      if npr.rand() < 1/self.tick:
        new_action = (0 if npr.rand() < 0.1 else 1)
      else:
        # if update_A:
        #   new_action = self.NN_A.forward(self.state_parser(state)).argmax() # new action
        # else:
        #   new_action = self.NN_B.forward(self.state_parser(state)).argmax() # new action

        new_action = self.NN_A.forward(self.state_parser(state)).argmax() # new action

    # update second neural network as needed
    if self.tick % 200 == 0:
      torch.save(self.NN_A, 'NN')
      self.NN_B = torch.load('NN')
      self.NN_B.eval()

    self.last_action = new_action # update state and action
    self.last_state = new_state

    return self.last_action

  def reward_callback(self, reward):
    '''This gets called so you can see what reward you get.'''

    self.last_reward = reward

  # def loss(self, pred_Q, target): # converges to immediately hitting top, has good intermediate results
  #   l = torch.mean((pred_Q - target)**2)
  #   return l

  def loss(self, pred_Q, target): # also converges to only immediately hitting top
    return (pred_Q - target)**2

  def state_parser(self, state):
    s = torch.tensor([state['tree']['dist'],
                    # state['tree']['top'],
                    state['tree']['bot'],
                    state['monkey']['vel'],
                    # state['monkey']['top'],
                    state['monkey']['bot']])
    return s

iters = 200
learner = Learner(alpha = 0.005, gamma = 0.7)
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




