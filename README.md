# torchgym

A PyTorch library that provides major RL algorithm(s) and functionalities for training OpenAI Gym agents.

> About

- torchgym provides RL algorithms built on top of PyTorch, specifically for OpenAI Gym environments.
- The library currently supports the DQN algorithm for Classic Control and Box-2D environments.
- The library currently supports the Colab environment to run the models.

> Installation

- Open a new Colab notebook.
- Run the following commands.
```
# Copy source code
!curl -L 'https://github.com/gitHubAndyLee2020/torchgym/archive/refs/heads/main.zip' -o torchgym.zip
!unzip torchgym.zip
!rm torchgym.zip
!mv torchgym-main torchgym
```
- To enable box-2d environments, run the following commands afterward.
```
# Install packages for box2d environments
!pip install swig
!pip install gym[box2d]
```

> Interface

> More

- To incorporate more RL algorithms, or to add support for MuJuCo, Toy Text, or Atari environments from OpenAI Gym, please submit a pull request with a simple proof of the code working and code snippets to replicate the code.
- You can find summaries of different RL algorithm implementations in the repos [RL Algorithm Summaries](https://github.com/gitHubAndyLee2020/OpenAI_Gym_RL_Algorithms_Database) and [Original RL Algorithm Implementations](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/tree/master).
