import torch
from torch import nn
from gym_env.env import Action
import numpy as np
class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='reinforce'):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True
        self.model = None
        self.action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}


    def initiate_agent(self, nb_actions):
        """initiate a deep Q agent"""
        # Initialize Neural Network
        self.model = torch.nn.Sequential(
            torch.nn.Linear(nb_actions,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,nb_actions),
            torch.nn.Softmax()
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        self.model.eval()
        prediction = self.model(torch.tensor(info).float())
        probs = np.array(prediction.detach().numpy())
        action = np.random.choice(len(self.action_space),p=probs)
        return action