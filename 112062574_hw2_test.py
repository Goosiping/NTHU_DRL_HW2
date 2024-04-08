import torch
import torch.nn as nn
import numpy as np

import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

'''
class QNetwork(nn.Module):
    """
    For RGB Image
    """
    def __init__(self, n_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out((3, 240, 256))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )  

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        out = self.fc(conv_out)
        return out
'''

class QNetwork(nn.Module):
    """
    For Gray Scale Image
    """
    def __init__(self, n_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out((1, 240, 256))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )  

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        out = self.fc(conv_out)
        return out

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.q = QNetwork(self.action_space.n)
        self.q.load_state_dict(torch.load('./112062574_hw2_data', map_location=torch.device('cpu')))
        print("INFO: Model loaded successfully.")
        

    def act(self, observation):
        observation = self._preprocess(observation)
        observation = torch.Tensor(observation.copy()).unsqueeze(0)
        with torch.no_grad():
            return self.q(observation).argmax().item()
        
    def _preprocess(self, state):
        state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
        state = state.astype(np.float32) / 255.0
        state = np.reshape(state, (1, 240, 256)) 
        return state       


if __name__ == "__main__":
    
    # Env
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    episodes = 1
    score = 0
    for e in range(episodes):
        agent = Agent()
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            env.render()
            score += reward

    print("INFO: Score: ", score)