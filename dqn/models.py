import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

# Define the neural network for Q-Learning
class Net(nn.Module):
    def __init__(self, num_state, num_action):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x)
        return action_value

# Define the DQN agent
class DQN():
    def __init__(self, num_state, num_action, capacity, learning_rate, memory_count, batch_size, gamma, update_count):
        super(DQN, self).__init__()
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.memory_count = memory_count
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_count = update_count
        self.target_net, self.act_net = Net(num_state, num_action), Net(num_state, num_action)
        self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.value_loss_log = []
        self.finish_step_log = []

    # Policy: Select action
    def select_action(self, state, num_action):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.act_net(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= 0.9:
            action = np.random.choice(range(num_action), 1).item()
        return action

    # Store transitions for experience replay
    def store_transition(self, transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1

    # Update Q-values and policy
    def update(self):
        if self.memory_count >= self.capacity:
            state = torch.tensor([t.state for t in self.memory]).float()
            action = torch.LongTensor([t.action for t in self.memory]).view(-1,1).long()
            reward = torch.tensor([t.reward for t in self.memory]).float()
            next_state = torch.tensor([t.next_state for t in self.memory]).float()
            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                target_v = reward + self.gamma * self.target_net(next_state).max(1)[0]
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                v = (self.act_net(state).gather(1, action))[index]
                loss = self.loss_func(target_v[index].unsqueeze(1), v)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.value_loss_log.append(loss.item())
                self.update_count += 1
                if self.update_count % 100 == 0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
        else:
            print("Memory Buff is too less")