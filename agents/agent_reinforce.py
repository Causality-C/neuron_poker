import sys
import numpy as np

from agents.reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN

class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN', load_model=None, env=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.dqn = None
        self.model = None
        self.env = env
        self.pi = None
        self.B = None

        if load_model:
            self.load(load_model)

    def initiate_agent(self, env):
        alpha = 3e-4
        baseline = False
        self.env = env

        """Initialize REINFORCE agent"""
        self.pi = PiApproximationWithNN(
            env.observation_space[0],
            env.action_space.n,
            alpha)

        if baseline:
            self.B = VApproximationWithNN(
                env.observation_space[0],
                alpha)
        else:
            self.B = Baseline(0.)


    def reinforce_train(self):
        gamma = 1.
        print(REINFORCE(self.env, gamma, 40, self.pi, self.B))