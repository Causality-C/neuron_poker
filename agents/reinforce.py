from typing import Iterable
import numpy as np

import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, state_dims, num_actions=0):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(state_dims, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 32))
        layers.append(nn.ReLU())

        if num_actions <= 0:
            layers.append(nn.Linear(32, 1))
        else:
            layers.append(nn.Linear(32, num_actions))
            layers.append(nn.Softmax(0))

        self.linear_relu_stack = nn.Sequential(*layers)


    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.state_dims = state_dims
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NeuralNetwork(state_dims, num_actions).to(device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.batch_size = 1

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size

    def __call__(self, legal_actions, s) -> int:
        self.model.eval()
        action_prob = np.nan_to_num(self.model(torch.from_numpy(s).float()).detach().numpy())
        # Fix for Duplicate Actions bug
        legal_actions = list(set(legal_actions))

        # Sum up legal action probabilities
        legal_actions_prob = 0
        for action in legal_actions:
            legal_actions_prob += action_prob[action.value]

        legal_probs = [action_prob[enum.value] for enum in legal_actions] / legal_actions_prob
        return legal_actions[np.random.choice(len(legal_actions), p=legal_probs)].value
        # choice = np.random.rand()
        # current_prob = 0
        # for a, probability in enumerate(set(legal_actions)):
        #     current_prob += probability/legal_actions_prob
        #     if choice < current_prob:
        #         return set(legal_actions)[a]
        #
        # return action_prob.size - 1

    def my_loss(self, output, target, action):
        # Find the expectation of the return times the gradient of the log policy
        # the loss is the inverse of this value

        # How do I find the gradient of the log policy? Is there some sort of gradient function?
        # A: The loss.backward function computes the gradient. All I need to do is find the log of the probabilities
        # times the return
        loss = torch.neg(torch.mul(torch.log(output[action]), target))
        return loss

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer


    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(torch.from_numpy(s).float())
        loss = self.my_loss(output, torch.tensor(delta * gamma_t).float(), a)
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.state_dims = state_dims
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NeuralNetwork(state_dims).to(device).float()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.batch_size = 1

    def __call__(self,s) -> float:
        self.model.eval()

        return self.model(torch.from_numpy(s).float()).detach().numpy()[0]

    def update(self,s,G):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(torch.from_numpy(s).float())
        loss = self.loss_function(output, torch.tensor(G).float())
        loss.backward()
        self.optimizer.step()

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    G_0 = np.zeros(num_episodes)
    for episodes in range(num_episodes):
        done = False
        state_list = []
        action_list = []
        reward_list = []
        s = env.reset()
        a = pi.__call__(env.legal_moves, s)

        # Generate an episode
        while not done:
            ns, r, done, info = env.step(a)
            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            s = ns
            a = pi.__call__(env.legal_moves,s)

        for t in range(len(state_list)):
            G = 0
            for k in range(len(state_list)-t):
                G += gamma**k * reward_list[t+k]
            s = state_list[t]
            a = action_list[t]
            error = G - V.__call__(s)
            V.update(s, G)
            pi.update(s, a, gamma ** t, error)
            if t == 0:
                G_0[episodes] = G
    return G_0

