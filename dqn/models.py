import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

# Define the neural network for Q-Learning
class Net(nn.Module):
    def __init__(self, num_state, num_action, net_layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()  # Initialize a ModuleList to hold all the layers
        
        # Input layer
        self.layers.append(nn.Linear(num_state, net_layers[0]))
        
        # Hidden layers
        for i in range(1, len(net_layers)):
            self.layers.append(nn.Linear(net_layers[i-1], net_layers[i]))
        
        # Output layer
        self.layers.append(nn.Linear(net_layers[-1], num_action))

    def forward(self, x):
        # Apply ReLU activation function to each layer except the last
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # No activation function for the last layer (action value layer)
        action_value = self.layers[-1](x)
        return action_value

# Define the DQN agent
class DQN():
    def __init__(self, num_state, num_action, learning_rate, gamma, exploration_rate, capacity, batch_size, net_layers):
        super(DQN, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory_count = 0
        self.update_count = 0
        self.target_net, self.act_net = Net(num_state, num_action, net_layers), Net(num_state, num_action, net_layers)
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
        if np.random.rand(1) >= (1 - self.exploration_rate):
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