# Flappy Bird AI - Deep Q-Learning with Double DQN and Epsilon-Greedy Strategy

This project is an AI agent designed to play the game **Flappy Bird** autonomously using **Deep Q-Learning**. It leverages a **Double Deep Q-Network (DDQN)** architecture, along with an **epsilon-greedy** exploration strategy to optimize decision-making in uncertain game environments. The project aims to demonstrate reinforcement learning principles by training an agent that learns to play Flappy Bird efficiently over time.

## Table of Contents
- [Introduction](#introduction)
- [Approach](#approach)
  - [Double Deep Q-Network (DDQN)](#double-deep-q-network-ddqn)
  - [Epsilon-Greedy Strategy](#epsilon-greedy-strategy)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Introduction

This project applies reinforcement learning techniques to train an AI agent capable of playing **Flappy Bird**. The agent is trained through a **reward-based mechanism**, where it receives positive rewards for staying alive longer in the game and negative rewards upon hitting obstacles or the ground. The goal is to maximize the cumulative rewards, enabling the agent to make better in-game decisions over time.

## Approach

### Double Deep Q-Network (DDQN)

The **Double Deep Q-Network (DDQN)** is an improvement over the standard DQN approach, reducing overestimation bias that can arise when selecting and evaluating actions based on the same network. DDQN uses two networks:
1. **Main Network**: Responsible for selecting the best actions.
2. **Target Network**: Used to evaluate the Q-value of the selected action.

This separation allows the agent to make more reliable updates, which leads to more stable training and improved performance.

### Soft Update for Target Network

To maintain stability during training, the **target network** is updated using a **soft update** method. Instead of completely replacing the target network weights with those from the main network, the soft update gradually blends them according to a parameter `tau`. This allows smoother transitions and helps avoid abrupt changes, which can destabilize training.

The update rule for each parameter is calculated using this equation used in the paper [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf): 
   ```math
   Q_{\text{target}} = \tau \cdot Q_{\text{current}} + (1 - \tau) \cdot Q_{\text{target}}
```

### Epsilon-Greedy Strategy

The **epsilon-greedy strategy** is a method to balance exploration and exploitation during training. The agent chooses random actions with probability `epsilon` to explore new strategies and chooses the best-known action with probability `1 - epsilon` for exploiting the learned strategies.

  - **Exploration**: Helps the agent discover new actions and strategies.
  - **Exploitation**: Allows the agent to use known strategies that have led to higher rewards.

The epsilon value decays over time, reducing the exploration rate as the agent gains experience and learns to make optimal choices.

## Model Architecture

The model is a simple neural network designed for the agent's decision-making:
    - **Input Layer**: Takes in features such as the bird’s horizontal and vertical distances to the nearest pipes and the bird’s current speed.
    - **Hidden Layer**: A fully connected dense layer with 256 nodes.
    - **Output Layer**: Outputs Q-values for two possible actions: `jump` and `no action`.

The model is optimized using the following hyperparameters:
    - **Learning Rate**: `0.0001`
    - **Discount Factor (gamma)**: `0.95`
    - **Batch Size**: `32`
    - **Memory Size**: `100000`
    - **Epsilon Decay Rate**: `0.998`
    - **Minimum Epsilon**: `0.05`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Flappy-Bird-AI-DDQN.git
   cd Flappy-Bird-AI-DDQN
   ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the game and train the model:
    ```bash
    Copy code
    python train.py
    ```


## Usage

1. Training the Model:
  - Modify hyperparameters in config.py if needed.
  - Run python train.py to start training the agent.

2. Testing the Model:
  - After training, run python test.py to observe the agent's performance in the Flappy Bird game.

## Results


## Future Improvements

Enhanced State Representation: Add additional features, such as distances to upcoming pipes, to help the agent make more informed decisions.
Prioritized Experience Replay: Implement experience prioritization to sample more valuable experiences for faster convergence.
Parameter Optimization: Experiment with network architecture and hyperparameters to improve learning efficiency and gameplay skill.
