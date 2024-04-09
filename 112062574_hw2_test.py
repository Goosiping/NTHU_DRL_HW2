import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque

import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

SHAPE = (4, 84, 84)

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
'''    
class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(SHAPE[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(SHAPE)

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

        self.frame_id = 0
        self.frame_skip = 4
        self.buffer = deque(maxlen = self.frame_skip)

    def act(self, observation):
        observation = self._preprocess(observation)
        if len(self.buffer) < self.frame_skip:
            while len(self.buffer) < self.frame_skip:
                self.buffer.append(observation)
        else:
            self.buffer.append(observation)
        
        if self.frame_id % self.frame_skip == 0:
            stacked = np.stack(self.buffer, axis=0)
            input_state = torch.FloatTensor(stacked.copy()).unsqueeze(0)
            actions_value = self.q(input_state)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            self.last_action = action
            self.frame_id += 1
            return action
        else:
            self.frame_id += 1
            return self.last_action
        
    def _preprocess(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (SHAPE[1], SHAPE[2]), interpolation=cv2.INTER_AREA)
        state = state.astype(np.float32) / 255.0
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