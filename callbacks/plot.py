import pickle
import matplotlib.pyplot as plt
import os

def plot(env_name, saved_model_id):
  # Create directory 'history' if it doesn't exist
  history_path = 'history'
  if not os.path.exists(history_path):
      os.makedirs(history_path)

  # Create subdirectory '<env_name>' inside of 'history' if it doesn't exist
  env_path = os.path.join(history_path, env_name)
  if not os.path.exists(env_path):
      os.makedirs(env_path)

  # Create subdirectory '<model_id>' inside of 'history/<env_name>' if it doesn't exist
  model_path = os.path.join(env_path, saved_model_id)
  if not os.path.exists(model_path):
      os.makedirs(model_path)

  value_loss_path = os.path.join(model_path, f'value_loss_log.pkl')
  finish_step_path = os.path.join(model_path, f'finish_step_log.pkl')
  value_loss_plot_path = os.path.join(model_path, f'value_loss_log_plot.png')
  finish_step_path = os.path.join(model_path, f'finish_step_log.pkl')
  finish_step_plot_path = os.path.join(model_path, f'finish_step_log_plot.png')

  # Load the data from the pickle files
  with open(value_loss_path, 'rb') as f:
      value_loss_log = pickle.load(f)

  with open(finish_step_path, 'rb') as f:
      finish_step_log = pickle.load(f)

  # Plotting the value_loss_log
  plt.figure(figsize=(12, 6))
  plt.plot(value_loss_log, label='Value Loss')
  plt.title('Value Loss Log')
  plt.xlabel('Update Count')
  plt.ylabel('Value Loss')
  plt.legend()
  plt.grid(True)
  # Save the plot as an image file
  plt.savefig(value_loss_plot_path)

  # Plotting the finish_step_log
  plt.figure(figsize=(12, 6))
  plt.plot(finish_step_log, label='Finish Step')
  plt.title('Finish Step Log')
  plt.xlabel('Episode')
  plt.ylabel('Finish Step')
  plt.legend()
  plt.grid(True)
  # Save the plot as an image file
  plt.savefig(finish_step_plot_path)
