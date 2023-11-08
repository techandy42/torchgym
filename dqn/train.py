import gym
import torch
import pickle
import json
import os
from collections import namedtuple
from .models import DQN
from .helper import get_eval_agent
from ..utils.save import save
from ..utils.time import parse_timedelta
from ..callbacks.record import record
from ..callbacks.plot import plot
from ..callbacks.eval import eval
import uuid
from datetime import datetime
import sys

def dqn_train(env_name, num_episodes, learning_rate=1e-3, gamma=0.995, exploration_rate=0.1, capacity=8000, batch_size=256, net_layers=[100], optimizer_label='Adam', optimizer_callback=None, loss_func_label='MSELoss', loss_func_callback=None, model_label=None, saved_model_id=None, callbacks=[]):
    try:
        if 'save_on_max_finish_step' in callbacks and 'save_on_min_finish_step' in callbacks:
            print('You cannot have both save_on_max_finish_step and save_on_min_finish_step callbacks.')
            print('Exiting from training...')
            return None

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
            optimizer_callback=optimizer_callback,
            loss_func_callback=loss_func_callback
        )

        # Load model weights and training history.
        if saved_model_id is not None:
            model_path = os.path.join('history', env_name, saved_model_id)
            weights_path = os.path.join(model_path, 'model_weights.pth')
            value_loss_path = os.path.join(model_path, 'value_loss_log.pkl')
            finish_step_path = os.path.join(model_path, 'finish_step_log.pkl')
            collected_reward_path = os.path.join(model_path, 'collected_reward_log.pkl')
            agent.act_net.load_state_dict(torch.load(weights_path))
            with open(value_loss_path, 'rb') as f:
                agent.value_loss_log = pickle.load(f)
            with open(finish_step_path, 'rb') as f:
                agent.finish_step_log = pickle.load(f)
            with open(collected_reward_path, 'rb') as f:
                agent.collected_reward_log = pickle.load(f)

        model_id = str(uuid.uuid4())
        model_saved = False

        agent.act_net.train()

        # Training loop.
        max_collected_reward = -sys.maxsize - 1
        for i_ep in range(num_episodes):
            state = env.reset()
            collected_reward = 0
            for t in range(10000):
                action = agent.select_action(state, num_action)
                next_state, reward, done, _, info = env.step(action)
                transition = Transition(state, action, reward, next_state)
                agent.store_transition(transition)
                state = next_state
                collected_reward += reward
                if done or t >= 9999:
                    # Update max_t and min_t 
                    if collected_reward > max_collected_reward:
                        max_collected_reward = collected_reward

                        if 'save_on_max_reward' in callbacks:
                            end_time = datetime.now()
                            training_duration = end_time - start_time
                            # Save model weights and training history.
                            save(
                                env_name=env_name,
                                model_id=model_id,
                                data={
                                    'model_label': model_id if model_label is None else model_label,
                                    'created_at': end_time.strftime("%Y-%m-%d %H:%M:%S") if saved_model_id is None else data['created_at'],
                                    'updated_at': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    'training_duration': str(training_duration) if saved_model_id is None else str(parse_timedelta(data['training_duration']) + training_duration),
                                    'env_name': env_name,
                                    'model_name': 'dqn',
                                    'model_id': model_id,
                                    'num_episodes': i_ep if data is None else i_ep + data['num_episodes'],
                                    'learning_rate': learning_rate,
                                    'gamma': gamma,
                                    'exploration_rate': exploration_rate,
                                    'capacity': capacity, 
                                    'batch_size': batch_size, 
                                    'net_layers': net_layers,
                                    'optimizer_label': optimizer_label,
                                    'loss_func_label': loss_func_label
                                }, 
                                agent=agent
                            )
                            print(f'Saving model weights on new maximum rewards at episode {i_ep} with total reward of {collected_reward}...')
                            model_saved = True

                    # Update agent and print training information.
                    agent.finish_step_log.append(t+1)
                    agent.collected_reward_log.append(collected_reward)
                    agent.update()
                    if i_ep % 10 == 0:
                        print("episodes {}, step is {}, total reward is {}".format(i_ep, t, collected_reward))
                    
                    break

        if not model_saved or 'save_on_max_reward' not in callbacks:
            end_time = datetime.now()
            training_duration = end_time - start_time
            # Save model weights and training history.
            save(
                env_name=env_name,
                model_id=model_id,
                data={
                    'model_label': model_id if model_label is None else model_label,
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
                    'optimizer_label': optimizer_label,
                    'loss_func_label': loss_func_label
                }, 
                agent=agent
            )
            print('Saving model on last episode...')

        # Initialize the evaluation DQN agent.
        eval_agent = get_eval_agent(
            env_name=env_name, 
            saved_model_id=model_id
        )

        # Handle callbacks.
        if 'record' in callbacks:
            record(
                env_name=env_name,
                agent=eval_agent,
                saved_model_id=model_id,
            )

        if 'plot' in callbacks:
            plot(
                env_name=env_name, 
                saved_model_id=model_id
            )

        eval(
            env_name=env_name,
            agent=eval_agent
        )

        print(f'Model saved to history/{env_name}/{model_id}')
        print(f'Finished training...')

        return model_id

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None