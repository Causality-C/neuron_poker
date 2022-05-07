import torch
from torch import nn
class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='qlearning'):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True
        self.model = None

    def initiate_agent(self, nb_actions):
        """initiate a deep Q agent"""

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use,unused-argument
        """Mandatory method that calculates the move based on the observation array and the action space."""
        # We shall not include action 3
        env, inf = (observation, info)  # not using the observation for random decision
        action = None

        # decide if explore or exploit

        # forward

        # save to memory

        # backward
        # decide what to use for training
        # update model
        # save weights

        return action