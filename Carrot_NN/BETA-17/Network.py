import torch
from torch import nn
from Hyperparams import NUM_STATES
from Hyperparams import NUM_ACTIONS
from Hyperparams import NODES



class Brain:

    def __init__(self):
        self.num_states = NUM_STATES
        self.num_actions = NUM_ACTIONS
        self.Q = None

    def modeling_NN(self):
        model = nn.Sequential()
        model.add_module('fc1', nn.Linear(self.num_states, NODES))
        model.add_module('relu1', nn.ReLU())
        model.add_module('fc2', nn.Linear(NODES, NODES))
        model.add_module('relu2', nn.ReLU())
        model.add_module('fc3', nn.Linear(NODES, self.num_actions))
        return model

    def action_order(self, state):
        '''Exploitation-이용'''
        self.Q.eval()
        with torch.no_grad():
            data = self.Q(state)
            action = torch.argmax(data).item()
        return action