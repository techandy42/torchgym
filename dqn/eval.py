import gym
import json
import os
from .models import DQN
from ..callbacks.eval import eval

def dqn_eval(env_name, saved_model_id):
    try:
        model_path = os.path.join('history', env_name, saved_model_id)
        data_path = os.path.join(model_path, 'data.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
            print('Evaluating existing model.')
            env_name = data['env_name']
            learning_rate = data['learning_rate']
            gamma = data['gamma']
            exploration_rate = data['exploration_rate']
            capacity = data['capacity']
            batch_size = data['batch_size']
            net_layers = data['net_layers']

        env = gym.make(env_name).unwrapped
        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n

        # Initialize the DQN agent.
        agent = DQN(
            num_state=num_state,
            num_action=num_action,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration_rate=exploration_rate,
            capacity=capacity, 
            batch_size=batch_size, 
            net_layers=net_layers,
            optimizer_callback=None,
            loss_func_callback=None
        )

        eval(
            env_name=env_name,
            agent=agent
        )

        return True
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False