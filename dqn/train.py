import gym
import torch
import pickle
from collections import namedtuple
from .models import DQN
from ..utils.save import save
from ..callbacks.record import record
from ..callbacks.plot import plot

def dqn_train(env_name, num_episodes, capacity=8000, learning_rate=1e-3, memory_count=0, batch_size=256, gamma=0.995, update_count=0, callbacks=[]):
    env = gym.make(env_name).unwrapped
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n
    
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

    agent = DQN(
        num_state=num_state,
        num_action=num_action,
        capacity=capacity, 
        learning_rate=learning_rate, 
        memory_count=memory_count, 
        batch_size=batch_size, 
        gamma=gamma, 
        update_count=update_count
    )

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

    # Save model weights and training history.
    save(
        env_name=env_name, 
        model_name='dqn', 
        hyperparameters={
            'num_episodes': num_episodes,
            'capacity': capacity, 
            'learning_rate': learning_rate, 
            'memory_count': memory_count, 
            'batch_size': batch_size, 
            'gamma': gamma, 
            'update_count': update_count
        }, 
        agent=agent
    )

    # Handle callbacks.
    if 'record' in callbacks:
        record(
            env_name=env_name,
            capacity=capacity,
            learning_rate=learning_rate,
            memory_count=memory_count,
            batch_size=batch_size,
            gamma=gamma,
            update_count=update_count
        )

    if 'plot' in callbacks:
        plot()