import os
import gym
import torch
from ..dqn.models import DQN

def eval(env_name, model_id, learning_rate, gamma, exploration_rate, capacity, batch_size, net_layers, optimizer):
    env = gym.make(env_name)
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n

    agent = DQN(
        num_state=num_state,
        num_action=num_action,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_rate=exploration_rate,
        capacity=capacity, 
        batch_size=batch_size, 
        net_layers=net_layers,
        optimizer=optimizer
    )

    # Construct the path to the weights file
    weights_path = os.path.join('history', env_name, model_id, 'model_weights.pth')
    agent.act_net.load_state_dict(torch.load(weights_path))

    total_num_steps = 0
    num_episodes = 10
    for i in range(num_episodes):
        state = env.reset()
        done = False
        num_steps = 0
        while not done:
            action = agent.select_action(state, num_action)
            state, reward, done, _ = env.step(action)
            num_steps += 1
        print(f'Number of steps in episode {i}: {num_steps}')
        total_num_steps += num_steps
    print(f'Total number of steps is: {total_num_steps}')
    avg_num_steps = total_num_steps / num_episodes
    print(f'Average number of steps is: {avg_num_steps}')

    env.close()
