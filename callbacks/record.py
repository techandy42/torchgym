import os
import gym
import imageio
import torch
from ..dqn.models import DQN

def record(env_name, model_id, capacity, learning_rate, memory_count, batch_size, gamma, update_count, video_filename='trained_model.mp4'):
    env = gym.make(env_name)
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n

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

    # Construct the path to the weights file
    weights_path = os.path.join('history', env_name, model_id, f'model_weights.pth')
    agent.act_net.load_state_dict(torch.load(weights_path))

    frames = []

    state = env.reset()
    done = False
    while not done:
        frames.append(env.render(mode="rgb_array"))
        action = agent.select_action(state, num_action)
        state, reward, done, _ = env.step(action)

    env.close()

    # Save the recorded frames as a video
    with imageio.get_writer(video_filename, fps=30) as video:
        for frame in frames:
            video.append_data(frame)
