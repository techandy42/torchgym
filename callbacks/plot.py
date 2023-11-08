import pickle
import matplotlib.pyplot as plt
import os

def plot(env_name, saved_model_id):
    # Directory structure setup
    history_path = 'history'
    env_path = os.path.join(history_path, env_name)
    model_path = os.path.join(env_path, saved_model_id)
    
    # Make sure the required directories exist
    os.makedirs(model_path, exist_ok=True)

    # Paths for the logs and plots
    value_loss_path = os.path.join(model_path, f'value_loss_log.pkl')
    finish_step_path = os.path.join(model_path, f'finish_step_log.pkl')
    collected_reward_path = os.path.join(model_path, f'collected_reward_log.pkl')
    
    value_loss_plot_path = os.path.join(model_path, f'value_loss_log_plot.png')
    finish_step_plot_path = os.path.join(model_path, f'finish_step_log_plot.png')
    collected_reward_plot_path = os.path.join(model_path, f'collected_reward_log_plot.png')

    # Load the data from the pickle files
    with open(value_loss_path, 'rb') as f:
        value_loss_log = pickle.load(f)

    with open(finish_step_path, 'rb') as f:
        finish_step_log = pickle.load(f)

    with open(collected_reward_path, 'rb') as f:
        collected_reward_log = pickle.load(f)

    # Plot value_loss_log
    plt.figure(figsize=(12, 6))
    plt.plot(value_loss_log, label='Value Loss')
    plt.title('Value Loss Log')
    plt.xlabel('Update Count')
    plt.ylabel('Value Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(value_loss_plot_path)
    plt.close()  # Close the figure to prevent it from showing in the next plot

    # Plot finish_step_log
    plt.figure(figsize=(12, 6))
    plt.plot(finish_step_log, label='Finish Step')
    plt.title('Finish Step Log')
    plt.xlabel('Episode')
    plt.ylabel('Finish Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(finish_step_plot_path)
    plt.close()  # Close the figure to prevent it from showing in the next plot

    # Plot collected_reward_log
    plt.figure(figsize=(12, 6))
    plt.plot(collected_reward_log, label='Collected Reward')
    plt.title('Collected Reward Log')
    plt.xlabel('Episode')
    plt.ylabel('Collected Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(collected_reward_plot_path)
    plt.close()  # Close the figure to free up memory
