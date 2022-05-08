import sys
import numpy as np
from matplotlib import pyplot as plt

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
        training_progress = REINFORCE(self.env, gamma, 1000, self.pi, self.B)
        sums = [0 for i in range(len(training_progress))]
        s = 0
        for i, val in enumerate(training_progress):
            s += val
            sums[i] = s

        fig, ax = plt.subplots()
        ax.plot(sums)
        plt.show()