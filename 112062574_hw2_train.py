import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import time
import cv2 as cv
import random
from collections import deque

import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

TARGET_SHAPE = (4, 84, 84)
STACK_FRAME = 4

class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(TARGET_SHAPE[0], 32, kernel_size=8, stride=4),
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

        conv_out_size = self._get_conv_out(TARGET_SHAPE)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        for i in range(0,8,2):
            self.conv[i].weight.data.normal_(0, 0.1)
        self.fc[1].weight.data.normal_(0, 0.1)
        self.fc[3].weight.data.normal_(0, 0.1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        out = self.fc(conv_out)
        return out
    
class DQN():
    def __init__(self,
        n_actions,
        input_shape,
        qnet,
        device,
        learning_rate = 2e-4,
        reward_decay = 0.99,
        replace_target_iter = 1000,
        memory_size = 30000,
        batch_size = 64
        ):

        # initialize parameters
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device
        self.learn_step_counter = 0
        self.init_memory()

        # Network
        self.qnet_eval = qnet(self.n_actions).to(self.device)
        self.qnet_target = qnet(self.n_actions).to(self.device)
        self.qnet_target.eval()
        self.optimizer = optim.RMSprop(self.qnet_eval.parameters(), lr=self.lr)

    def choose_action(self, state, epsilon = 0):
        input_state = torch.FloatTensor(state.copy()).unsqueeze(0).to(self.device)
        actions_value = self.qnet_eval.forward(input_state)
        if np.random.uniform() > epsilon:   # greedy
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:   # random
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):

        if self.learn_step_counter % self.replace_target_iter == 0 :
            self.qnet_target.load_state_dict(self.qnet_eval.state_dict())

        if self.memory_counter > self.memory_size :
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else :
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        b_s = torch.FloatTensor(self.memory["s"][sample_index]).to(self.device)
        b_a = torch.LongTensor(self.memory["a"][sample_index]).to(self.device)
        b_r = torch.FloatTensor(self.memory["r"][sample_index]).to(self.device)
        b_s_ = torch.FloatTensor(self.memory["s_"][sample_index]).to(self.device)
        b_d = torch.FloatTensor(self.memory["done"][sample_index]).to(self.device)

        q_curr_eval = self.qnet_eval(b_s).gather(1, b_a)
        q_next_target = self.qnet_target(b_s_).detach()
        q_next_eval = self.qnet_eval(b_s_).detach()
        next_state_values = q_next_target.gather(1, q_next_eval.max(1)[1].unsqueeze(1))
        q_curr_recur = b_r + self.gamma * next_state_values

        self.loss = F.smooth_l1_loss(q_curr_eval, q_curr_recur)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        return self.loss.detach().cpu().numpy()


    def init_memory(self):
        self.memory = {
            "s": np.zeros((self.memory_size, *self.input_shape)),
            "a": np.zeros((self.memory_size, 1)),
            "r": np.zeros((self.memory_size, 1)),
            "s_": np.zeros((self.memory_size, *self.input_shape)),
            "done": np.zeros((self.memory_size, 1)),
        }

    def store_transition(self, s, a, r, s_, d):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        if self.memory_counter <= self.memory_size:
            index = self.memory_counter % self.memory_size
        else:
            index = np.random.randint(self.memory_size)
        self.memory["s"][index] = s
        self.memory["a"][index] = a
        self.memory["r"][index] = r
        self.memory["s_"][index] = s_
        self.memory["done"][index] = d
        self.memory_counter += 1
    
    def save_model(self):
        torch.save(self.qnet_eval.state_dict(), "./112062574_hw2_data")

def epsilon_compute(frame_id, epsilon_max=1, epsilon_min=0.05, epsilon_decay=100000):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-frame_id / epsilon_decay)

class MarioWrapper(gym.Wrapper):
    def __init__(self, env, image_shape):
        super().__init__(env)
        self.k = STACK_FRAME
        self.image_shape = image_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=image_shape, dtype=np.float32)

    def _preprocess(self, state):
        state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
        state = cv.resize(state, (self.image_shape[1], self.image_shape[2]), interpolation=cv.INTER_AREA)
        state = state.astype(np.float32) / 255.0
        return state
    
    def step(self, action):
        state_next = []
        reward = 0
        done = False
        for i in range(self.k) :
            if not done :
                state_next_f, reward_f, done_f, info_f = self.env.step(action)
                state_next_f = self._preprocess(state_next_f)
                reward += reward_f
                done = done_f
                info = info_f
            state_next.append(state_next_f[np.newaxis, ...])
        state_next = np.concatenate(state_next, 0)
        
        return state_next, reward, done, info
    
    def reset(self):
        state = self.env.reset()
        state = self._preprocess(state)
        state = state[np.newaxis, ...].repeat(self.k, axis=0)
        return state


if __name__ == "__main__":

    # Environment
    env_origin = gym_super_mario_bros.make('SuperMarioBros-v0')
    env_origin = JoypadSpace(env_origin, COMPLEX_MOVEMENT)
    env = MarioWrapper(env_origin, (1, 84, 84))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQN(
                n_actions = gym.spaces.Discrete(12).n,
                input_shape = TARGET_SHAPE,
                qnet = QNetwork,
                device = device,
                learning_rate = 2e-4, 
                reward_decay = 0.99,
                replace_target_iter = 1000, 
                memory_size = 30000,
                batch_size = 64,
    )

    episodes = 500
    sample_episodes = 2
    total_step = 0
    start_time = time.time()
    print("INFO: Start training")
    for episode in range(episodes):

        # Reset environment.
        state = env.reset()
        state, reward, done, info = env.step(0)

        # Initialize information.
        step = 0
        total_reward = 0
        my_total_reward = 0
        loss = 0
        frame_id = 0

        # Environment info
        MARIO ={"small": 0, "tall": 1, "fireball": 2}
        prev_creatures = info["life"]
        prev_mario = info["status"]
        x_pos = deque(maxlen = 8)
        x_pos.append(info["x_pos"])
        while True:
            
            # Training Data Collection
            epsilon = epsilon_compute(total_step)
            if episode < sample_episodes:
                if random.random() < epsilon:
                    action = agent.choose_action(state)
                else:
                    action = int(np.random.choice([4, 3, 0, 9], 1, p = [0.7, 0.1, 0.1, 0.1]))
            else:
                action = agent.choose_action(state, epsilon)

            # Get next stacked state.
            state_next, reward, done, info = env.step(action)

            # Reward Engineering
            my_reward = reward
            # 1. Kill Creatures
            if info["life"] < prev_creatures: 
                my_reward += 3
            # 2. Mario status
            if MARIO[info["status"]] > MARIO[prev_mario]:
                my_reward += 5
            # 3. X position
            x_pos.append(info["x_pos"])
            if len(set(x_pos)) < 2:
                my_reward -= 0.1

            prev_creatures = info["life"]
            prev_mario = info["status"]

            agent.store_transition(state, action, my_reward, state_next, done)

            if total_step > 4 * agent.batch_size :
                loss = agent.learn()

            state = state_next.copy()
            frame_id += 1
            step += 1
            total_step += 1
            total_reward += reward
            my_total_reward += my_reward

            if total_step % 100 == 0 or done:
                print('\rEpisode: {:3d} | Step: {:3d} / {:3d} | Reward: {:.3f} / {:.3f} | My_reward: {:.3f} / {:.3f} | Loss: {:.3f} | Epsilon: {:.3f} | Time:{:.3f}'\
                    .format(episode, step, total_step, reward, total_reward, my_reward, my_total_reward, loss, epsilon, time.time() - start_time), end="")
            
            if total_step % 10000 == 0:
                agent.save_model()

            if done:
                print()
                break
    
    agent.save_model()
    print("INFO: Training finished.")
        