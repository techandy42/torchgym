import gym
import torch
import pickle
from collections import namedtuple
from .models import DQN

def dqn_train(env_name='CartPole-v0', num_episodes=100000, capacity=8000, learning_rate=1e-3, memory_count=0, batch_size=256, gamma=0.995, update_count=0):
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

    # Save the model weights
    torch.save(agent.act_net.state_dict(), 'dqn_model_weights.pth')
    with open('dqn_value_loss_log.pkl', 'wb') as f:
        pickle.dump(agent.value_loss_log, f)
    with open('dqn_finish_step_log.pkl', 'wb') as f:
        pickle.dump(agent.finish_step_log, f)