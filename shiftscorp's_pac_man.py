

!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!pip install ale-py
!apt-get install -y swig
!pip install gymnasium[box2d]


import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

class Network(nn.Module):

  def __init__(self, action_size, seed = 42):
    super(Network, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1)
    self.bn4 = nn.BatchNorm2d(128)
    self.fc1 = nn.Linear(10 * 10 * 128, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, action_size)

  def forward(self, state):
    x = F.relu(self.bn1(self.conv1(state)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x))) 
    x = F.relu(self.bn4(self.conv4(x)))
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)
      
import ale_py
import gymnasium as gym
env = gym.make('MsPacmanNoFrameskip-v0', full_action_space = False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99

from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
  frame = Image.fromarray(frame)
  preprocess = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
  return preprocess(frame).unsqueeze(0)

from threading import local
class Buddy():
  def __init__(self, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.local_qnet = Network(action_size).to(self.device)
    self.target_qnet = Network(action_size).to(self.device)
    self.optimizer = optim.Adam(self.local_qnet.parameters(), lr = learning_rate)
    self.memory = deque(maxlen = 10000)

  def step(self, state, action, reward, next_state, done):
    state = preprocess_frame(state)
    next_state = preprocess_frame(next_state)
    self.memory.append((state, action, reward, next_state, done))
    if len(self.memory) > minibatch_size:
      experiences = random.sample(self.memory, k = minibatch_size)
      self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.):
    state = preprocess_frame(state).to(self.device)
    self.local_qnet.eval()
    with torch.no_grad():
      action_values = self.local_qnet(state)
    self.local_qnet.train()
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    states, actions, rewards, next_states, dones = zip(*experiences)
    states  = torch.from_numpy(np.vstack(states)).float().to(self.device)
    actions  = torch.from_numpy(np.vstack(actions)).long().to(self.device)
    rewards  = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
    next_states  = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
    dones  = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
    next_q_targets = self.target_qnet(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + (discount_factor * next_q_targets * (1-dones))
    q_expected = self.local_qnet(states).gather(1, actions)
    loss = F.mse_loss(q_expected, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

agent = Buddy(number_actions)

number_episodes = 500
maximum_num_steps_per_ep = 2000
epsilon_start_val = 1.0
epsilon_end_val = 0.01
epsilon_decay_val = 0.99
epsilon = epsilon_start_val
scores_on100_ep = deque(maxlen = 100)

for episode in range(1,number_episodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(maximum_num_steps_per_ep):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores_on100_ep.append(score)
  epsilon = max(epsilon_end_val, epsilon_decay_val * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on100_ep)), end = "")
  if episode % 100 == 0:
     print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on100_ep)))
  if np.mean(scores_on100_ep) >= 500.0:
     print('\nEnvironment solved in {:d} episodes!\Average Score: {:.2f}'.format(episode, np.mean(scores_on100_ep)))
     torch.save(agent.local_qnet.state_dict(), 'checkpoint.pth')
     break

import glob
import io
import base64
import imageio
from IPython.display import HTML, display

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'MsPacmanNoFrameskip-v0')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()
