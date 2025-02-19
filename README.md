# Artificial Intelligence for Classic Snake Game


## Introduction
This project focuses on developing a reinforcement learning-based artificial intelligence (AI) for the classic Snake game. The AI is trained using Deep Q-Learning (DQL) to play Snake autonomously. The objective is to create an AI that learns from its environment and improves its performance over time. This project explores the integration of reinforcement learning with game development and addresses the challenges involved.

## Analysis of the Task

### Possible Approaches to Solve the Problem
Several approaches can be used to train an AI to play Snake:

- **Rule-Based Approach**: The snake follows predefined rules to avoid obstacles and chase food. This method is simple but lacks adaptability.
- **Heuristic-Based Approach**: Algorithms like A* help the snake find the shortest path to the food. While better than the rule-based approach, it still lacks learning capability.
- **Machine Learning Approach (Reinforcement Learning)**: A deep Q-network (DQN) is used to train the AI. This allows for:
  - **Adaptability**: The AI can learn and adjust to new situations.
  - **Performance Improvement**: The AI improves over time.
  - **Generalization**: The model can handle unseen game states.

#### Challenges:
- **Computationally Intensive**: Requires significant resources.
- **Complex Implementation**: DQN is more difficult to set up than rule-based methods.
- **Training Time**: Learning can be slow, especially for large state spaces.

### Data Representation
Unlike supervised learning, reinforcement learning generates data dynamically during gameplay. The key data components are:

- **State**: The game configuration (snake’s position, direction, food location).
- **Action**: The AI’s decision (move left, right, or straight).
- **Reward**: Feedback (positive for eating food, negative for collisions).

## Tools and Libraries

- **Pygame**: Used for game implementation and rendering.
- **PyTorch**: Used for building and training the deep Q-network.
- **NumPy**: Used for numerical operations in Q-learning.
- **Matplotlib**: Used to visualize training progress.

## Software Architecture

### Classes and Their Implementation

- **SnakeGameAI**: Manages the game state, handles actions, and updates rewards.
- **DeepQNetwork**: Defines the neural network architecture for Q-value approximation.
- **NeuralNetworkTrainer**: Manages the training process, including loss function and optimization.
- **Agent**: Stores experiences, selects actions, and trains the model using an epsilon-greedy policy.

### External Specification

- **AI Training and Execution**: Run `train_snake_ai.py` to train the AI. Execute `Snake_AI.py` to play autonomously.
- **Visualization**: The script `plot_utils.py` generates training performance graphs.

## Experiments and Results

### Training Parameters
- **Memory Size**: `MAX_MEMORY = 100,000`
- **Batch Size**: `BATCH_SIZE = 1000`
- **Learning Rate**: `LR = 0.001`
- **Discount Factor**: `gamma = 0.9`

### Performance Metrics
- **After 290 games**:
  - **Average Score**: 21.16
  - **Highest Score**: 86
- **After modifying the reward system (food proximity bonus)**:
  - **After 248 games**:
  - **Average Score**: 24
  - **Highest Score**: 76

### Observations
- The AI effectively avoids walls but sometimes gets trapped in loops caused by its own body.
- Reward function design significantly impacts learning effectiveness.
- 

## Author
Dawid Kosiński
