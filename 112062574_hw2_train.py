import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import time

import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class QNetwork(nn.Module):
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
    
class DQN():
    def __init__(self,
        n_actions,
        input_shape,
        qnet,
        device,
        learning_rate = 2e-4,
        reward_decay = 0.99,
        replace_target_iter = 1000,
        memory_size = 10000,
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
        self.optimizer = optim.Adam(self.qnet_eval.parameters(), lr=self.lr)

    def choose_action(self, state, epsilon=0):
        input_state = torch.FloatTensor(state.copy()).unsqueeze(0).to(self.device)
        actions_value = self.qnet_eval.forward(input_state)
        if np.random.uniform() > epsilon:   # greedy
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:   # random
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # TODO(Lab-5): DQN core algorithm.
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
        self.image_shape = image_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=image_shape, dtype=np.float32)

    def _preprocess(self, state):
        state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
        state = state.astype(np.float32) / 255.0
        state = np.reshape(state, self.image_shape)
        return state
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self._preprocess(state)
        return state, reward, done, info
    
    def reset(self):
        state = self.env.reset()
        state = self._preprocess(state)
        return state


if __name__ == "__main__":

    # Environment
    env_origin = gym_super_mario_bros.make('SuperMarioBros-v0')
    env_origin = JoypadSpace(env_origin, COMPLEX_MOVEMENT)
    env = MarioWrapper(env_origin, (1, 240, 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQN(
                n_actions = gym.spaces.Discrete(12).n,
                input_shape = (1, 240, 256),
                qnet = QNetwork,
                device = device,
                learning_rate = 2e-4, 
                reward_decay = 0.99,
                replace_target_iter = 1000, 
                memory_size = 10000,
                batch_size = 64,
    )

    episodes = 200
    total_step = 0
    start_time = time.time()
    print("INFO: Start training")
    for episode in range(episodes):

        # Reset environment.
        state = env.reset()

        # Initialize information.
        step = 0
        total_reward = 0
        loss = 0

        # One episode.
        while True:
            # TODO(Lab-6): Select action.
            epsilon = epsilon_compute(total_step)
            action = agent.choose_action(state, epsilon)

            # Get next stacked state.
            state_next, reward, done, info = env.step(action)

            # TODO(Lab-7): Train RL model.
            agent.store_transition(state, action, reward, state_next, done)
            if total_step > 4*agent.batch_size :
                loss = agent.learn()

            state = state_next.copy()
            step += 1
            total_step += 1
            total_reward += reward

            if total_step % 100 == 0 or done:
                print('\rEpisode: {:3d} | Step: {:3d} / {:3d} | Reward: {:.3f} / {:.3f} | Loss: {:.3f} | Epsilon: {:.3f} | Time:{:.3f}'\
                    .format(episode, step, total_step, reward, total_reward, loss, epsilon, time.time() - start_time), end="")
            
            if total_step % 10000 == 0:
                agent.save_model()

            if done:
                print()
                break
    
    agent.save_model()
    print("INFO: Training finished.")
        