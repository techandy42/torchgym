import os
import gym

def eval(env_name, agent):
    env = gym.make(env_name)

    # Ensure the agent's network is in evaluation mode
    agent.act_net.eval()
    
    total_num_steps = 0
    num_episodes = 10
    for i in range(num_episodes):
        state = env.reset()
        done = False
        num_steps = 0
        while not done:
            # Assuming select_action method does not perform any learning updates
            action = agent.select_action(state, exploration=False)  # Disable exploration if necessary
            state, reward, done, _ = env.step(action)
            num_steps += 1
        print(f'Number of steps in episode {i}: {num_steps}')
        total_num_steps += num_steps
    avg_num_steps = total_num_steps / num_episodes
    print(f'Average number of steps per episode: {avg_num_steps}')

    env.close()
