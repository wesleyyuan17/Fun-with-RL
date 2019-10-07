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
    self.hidden_dim = 8
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
    self.pred_NN = Neural_Network()
    self.target_NN = Neural_Network()
    self.optimizer = torch.optim.Adam(self.pred_NN.parameters(), lr=alpha)
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
      '''Failed implementation of randomly selecting a NN to update, go with original idea of updating
         every certain number of iterations'''

      # if npr.rand() < 0.5: # pick one NN to update

      #   update_A = True # track which NN is being updated

      #   pred_Q = self.pred_NN(self.last_state)[self.last_action]
      #   a_star = self.pred_NN(new_state).argmax()
      #   target = self.last_reward + self.gamma*self.target_NN(new_state)[a_star]
      # else:

      #   update_A = False

      #   pred_Q = self.target_NN(self.last_state)[self.last_action]
      #   a_star = self.target_NN(new_state).argmax()
      #   target = self.last_reward + self.gamma*self.pred_NN(new_state)[a_star]

      a_star = self.pred_NN(new_state).argmax()
      target = self.last_reward + self.gamma*self.target_NN(new_state)[a_star]
      self.optimizer.zero_grad() # zero gradient buffer (order s.t. gradient step only based on prediction)

      pred_Q = self.pred_NN(self.last_state)[self.last_action]

      loss = self.loss(pred_Q, target.item())
      self.loss_val.append(loss)
      # print(loss.item())
      loss.backward() # set up gradients in buffer
      print(self.pred_NN.fc1.bias.grad)
      self.optimizer.step() # perform update

      self.tick += 1 # track iterations

      # take action based on epsilon-greedy
      if npr.rand() < max(1/self.tick, 0.01):
        new_action = (0 if npr.rand() < 0.1 else 1)
      else:
        # if update_A:
        #   new_action = self.pred_NN.forward(self.state_parser(state)).argmax() # new action
        # else:
        #   new_action = self.target_NN.forward(self.state_parser(state)).argmax() # new action

        new_action = self.pred_NN.forward(self.state_parser(state)).argmax() # new action

    # update second neural network as needed
    if self.tick % 200 == 0:
      torch.save(self.pred_NN, 'NN')
      self.target_NN = torch.load('NN')
      self.target_NN.eval()

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

iters = 500
learner = Learner(alpha = 0.01, gamma = 0.7)
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




