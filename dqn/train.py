import gym
import torch
import pickle
import json
import os
from collections import namedtuple
from .models import DQN
from ..utils.save import save
from ..callbacks.record import record
from ..callbacks.plot import plot
from ..callbacks.eval import eval
import uuid
from datetime import datetime, timedelta

def parse_timedelta(delta_str):
    try:
        # Initialize days and microseconds to zero
        days = 0
        microseconds = 0

        # Check if the string includes days
        if ', ' in delta_str:
            days_str, time_str = delta_str.split(', ')
            days = int(days_str.split()[0])  # Extract the number of days
        else:
            time_str = delta_str

        # Check if the string includes microseconds
        if '.' in time_str:
            time_str, microseconds_str = time_str.split('.')
            microseconds = int(microseconds_str)

        # Extract hours, minutes, seconds from the time part
        hours, minutes, seconds = [int(part) for part in time_str.split(':')]

        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
    except Exception as e:
        print('timedelta parsing failed...')
        return timedelta(days=0, hours=0, minutes=0, seconds=0, microseconds=0)

def dqn_train(env_name, num_episodes, learning_rate=1e-3, gamma=0.995, exploration_rate=0.1, capacity=8000, batch_size=256, net_layers=[100], optimizer_label='Adam', optimizer_callback=None, saved_model_id=None, callbacks=[]):
    try:
        start_time = datetime.now()

        # Load hyperparameters from saved model.
        data = None
        if saved_model_id is not None:
            model_path = os.path.join('history', env_name, saved_model_id)
            data_path = os.path.join(model_path, 'data.json')
            with open(data_path, 'r') as f:
                data = json.load(f)
                print('Note that env_name and net_layers will be overriden with the values from the saved model.')
                env_name = data['env_name']
                net_layers = data['net_layers']
        
        env = gym.make(env_name).unwrapped
        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n
        
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

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
            optimizer_callback=optimizer_callback
        )

        # Load model weights and training history.
        if saved_model_id is not None:
            model_path = os.path.join('history', env_name, saved_model_id)
            weights_path = os.path.join(model_path, 'model_weights.pth')
            value_loss_path = os.path.join(model_path, 'value_loss_log.pkl')
            finish_step_path = os.path.join(model_path, 'finish_step_log.pkl')
            agent.act_net.load_state_dict(torch.load(weights_path))
            with open(value_loss_path, 'rb') as f:
                agent.value_loss_log = pickle.load(f)
            with open(finish_step_path, 'rb') as f:
                agent.finish_step_log = pickle.load(f)

        agent.act_net.train()

        # Training loop.
        for i_ep in range(num_episodes):
            state = env.reset()
            for t in range(10000):
                action = agent.select_action(state, num_action)
                next_state, reward, done, _, info = env.step(action)
                transition = Transition(state, action, reward, next_state)
                agent.store_transition(transition)
                state = next_state
                if done or t >= 9999:
                    agent.finish_step_log.append(t+1)
                    agent.update()
                    if i_ep % 10 == 0:
                        print("episodes {}, step is {} ".format(i_ep, t))
                    break

        model_id = str(uuid.uuid4())

        end_time = datetime.now()

        training_duration = end_time - start_time

        # Save model weights and training history.
        save(
            env_name=env_name,
            model_id=model_id,
            data={
                'created_at': end_time.strftime("%Y-%m-%d %H:%M:%S") if saved_model_id is None else data['created_at'],
                'updated_at': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                'training_duration': str(training_duration) if saved_model_id is None else str(parse_timedelta(data['training_duration']) + training_duration),
                'env_name': env_name,
                'model_name': 'dqn',
                'model_id': model_id,
                'num_episodes': num_episodes if data is None else num_episodes + data['num_episodes'],
                'learning_rate': learning_rate,
                'gamma': gamma,
                'exploration_rate': exploration_rate,
                'capacity': capacity, 
                'batch_size': batch_size, 
                'net_layers': net_layers,
                'optimizer_label': optimizer_label
            }, 
            agent=agent
        )

        # Handle callbacks.
        if 'record' in callbacks:
            record(
                env_name=env_name,
                agent=agent,
                model_id=model_id,
            )

        if 'plot' in callbacks:
            plot(
                env_name=env_name, 
                model_id=model_id
            )

        eval(
            env_name=env_name,
            agent=agent
        )

        print(f'Model saved to history/{env_name}/{model_id}')
        print(f'Finished training...')

        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False