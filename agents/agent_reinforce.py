import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
import logging

from agents.reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN, NeuralNetwork

logger = logging.getLogger(__name__)

class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN', load_model=None, env=None):
        """Initialization of an agent"""
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
            self.reinforce_load(load_model, env)

    def initiate_agent(self, env):
        alpha = 3e-4
        baseline = False
        self.env = env

        """Initialize REINFORCE agent"""
        self.pi = PiApproximationWithNN(
            env.observation_space[0],
            env.action_space.n,
            alpha)

        self.B = VApproximationWithNN(
            env.observation_space[0],
            alpha)



    def reinforce_train(self, env_name):
        gamma = 1.
        # List the amount won/loss during each poker game
        training_progress = REINFORCE(self.env, gamma, 500, self.pi, self.B)

        # Save the generated model
        torch.save({
            'pi_state_dict' : self.pi.get_model().state_dict(),
            'value_state_dict' : self.B.get_model().state_dict(),
            'pi_optimizer_dict' : self.pi.get_optimizer().state_dict(),
            'value_optimizer_dict' : self.B.get_optimizer().state_dict()
        }, env_name)

        # Plot the total amount of winnings/losses from 1000 episodes
        sums = [0 for i in range(len(training_progress))]
        s = 0
        for i, val in enumerate(training_progress):
            s += val
            sums[i] = s

        fig, ax = plt.subplots()
        ax.plot(sums)
        plt.show()

    def reinforce_load(self, model_name, env):
        # Initialize the agent
        self.initiate_agent(env)

        # Loads the model
        checkpoint = torch.load(model_name)
        self.pi.get_model().load_state_dict('pi_state_dict')
        self.B.get_model().load_state_dict(checkpoint['value_state_dict'])
        self.pi.get_optimizer().load_state_dict('pi_optimizer_dict')
        self.B.get_optimizer().load_state_dict(checkpoint['value_optimizer_dict'])

        # Set mode to evaluation
        self.pi.get_model().eval()
        self.B.get_model().eval()