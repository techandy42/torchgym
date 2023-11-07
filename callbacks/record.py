import os
import gym
import imageio

def record(env_name, agent, model_id):
    env = gym.make(env_name)
    
    # Ensure the agent's network is in evaluation mode
    agent.act_net.eval()

    frames = []
    state = env.reset()
    done = False
    while not done:
        frames.append(env.render(mode="rgb_array"))
        action = agent.select_action(state, exploration=False)  # Disable exploration if necessary
        state, reward, done, _ = env.step(action)

    env.close()

    video_path = os.path.join('history', env_name, model_id, 'trained_model.mp4')

    # Save the recorded frames as a video
    with imageio.get_writer(video_path, fps=30) as video:
        for frame in frames:
            video.append_data(frame)
