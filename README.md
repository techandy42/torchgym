# torchgym

A PyTorch library that provides major RL algorithm(s) and functionalities for training OpenAI Gym agents.

### About

- torchgym provides RL algorithms built on top of PyTorch, specifically for OpenAI Gym environments.
- The library currently supports the DQN algorithm for Classic Control and Box-2d environments, any valid environment can be implemented by simply switching the environment name.
- The library currently supports the Colab environment to run the models.
- To learn more about OpenAI Gym, please visit the [official website](https://gymnasium.farama.org/).

### Installation

- Open a new Colab notebook.
- Run the following commands.
```
# Copy source code
!curl -L 'https://github.com/gitHubAndyLee2020/torchgym/archive/refs/heads/main.zip' -o torchgym.zip
!unzip torchgym.zip
!rm torchgym.zip
!mv torchgym-main torchgym
```
- To enable Box-2d environments, run the following commands afterward.
```
# Install packages for box2d environments
!pip install swig
!pip install gym[box2d]
```

### Structure

- The library code will be located in the folder `torchgym`, where the submodules can be imported for usage.
- The submodule `dqn` contains all the functions for training and using DQN model agents.
- The submodule `functions` contains all the helper functions.

### Training Model

> `dqn_train` Function Specification

```
# The default hyperparameters are optimized for the MountainCar-v0 environment. 
def dqn_train(
  env_name: str, # a valid OpenAI gym environment name from Classical Control or Box-2d
  num_episodes: int, # the number of training episodes 
  episode_length: int, # (default: 10000) the number of steps in each episode, set this to the End of Episode number specified at the OpenAI Gymnasium website
  learning_rate: float, # (default: 1e-3) the rate at which the neural network is updated
  gamma: float, # (default: 0.995) the discount rate
  exploration_rate: float, # (default: 0.1) the probability of the model choosing random action during training 
  capacity: int, # (default: 8000) the number of experiences stored before starting training the model
  batch_size: int, # (default: 256) the batch size of the training data
  net_layers: int[], # (default: [100]) the specification of the hidden neural network shape for the Actor Network
  optimizer_label: str, # (default: 'Adam') the name of the optimizer used, doesn't affect the training
  optimizer_callback: None | callback, # (default: None) more on this below
  loss_func_label: str, # (default: 'MSELoss') the name of the loss function used, doesn't affect the training
  loss_func_callback: None | callback, # more on this below
  model_label: None | str, # (default: None) the name of the model, doesn't affect the training
  saved_model_id: None | str, # (default: None) more on this later
  callbacks: str[], # (default: []) if 'record', 'plot', or 'save_on_max_reward' are included, these callbacks are called during training, more on this later
):
  ...
```

> Example
```python
from torchgym.dqn.train import dqn_train

model_id = dqn_train(env_name='MountainCar-v0', num_episodes=1000, episode_length=200, model_label='model1', callbacks=['record', 'plot', 'save_on_max_reward'])
```

### More

- Please refer to `torchgym_dqn.ipynb` for a comprehensive overview of using the library.
- To incorporate more RL algorithms, or to add support for MuJuCo, Toy Text, or Atari environments from OpenAI Gym, please submit a pull request with a simple proof of the code working and code snippets to replicate the code.
- You can find summaries of different RL algorithm implementations in the repos [RL Algorithm Summaries](https://github.com/gitHubAndyLee2020/OpenAI_Gym_RL_Algorithms_Database) and [Original RL Algorithm Implementations](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/tree/master).

> Note

- The `dqn` module is only for environments with discrete action space.
