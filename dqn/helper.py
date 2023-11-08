import gym
import torch
import pickle
import json
import os
from .models import DQN

def get_eval_agent(env_name, saved_model_id):
    env = gym.make(env_name).unwrapped
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n

    model_path = os.path.join('history', env_name, saved_model_id)
    data_path = os.path.join(model_path, 'data.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    eval_agent = DQN(
        num_state=num_state,
        num_action=num_action,
        learning_rate=data['learning_rate'],
        gamma=data['gamma'],
        exploration_rate=data['exploration_rate'],
        capacity=data['capacity'], 
        batch_size=data['batch_size'], 
        net_layers=data['net_layers'],
        optimizer_callback=None,
        loss_func_callback=None
    )

    # Load model weights and training history.
    weights_path = os.path.join(model_path, 'model_weights.pth')
    value_loss_path = os.path.join(model_path, 'value_loss_log.pkl')
    finish_step_path = os.path.join(model_path, 'finish_step_log.pkl')
    collected_reward_path = os.path.join(model_path, 'collected_reward_log.pkl')
    eval_agent.act_net.load_state_dict(torch.load(weights_path))
    with open(value_loss_path, 'rb') as f:
        eval_agent.value_loss_log = pickle.load(f)
    with open(finish_step_path, 'rb') as f:
        eval_agent.finish_step_log = pickle.load(f)
    with open(collected_reward_path, 'rb') as f:
        eval_agent.collected_reward_log = pickle.load(f)

    return eval_agent