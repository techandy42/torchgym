import pickle
import matplotlib.pyplot as plt

def plot():
  # Load the data from the pickle files
  with open('dqn_value_loss_log.pkl', 'rb') as f:
      value_loss_log = pickle.load(f)

  with open('dqn_finish_step_log.pkl', 'rb') as f:
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
  plt.savefig('dqn_value_loss_log.png')

  # Plotting the finish_step_log
  plt.figure(figsize=(12, 6))
  plt.plot(finish_step_log, label='Finish Step')
  plt.title('Finish Step Log')
  plt.xlabel('Episode')
  plt.ylabel('Finish Step')
  plt.legend()
  plt.grid(True)
  # Save the plot as an image file
  plt.savefig('dqn_finish_step_log.png')
