import random
import torch
import numpy as np
from plot_utils import plot
from ai_model import DeepQNetwork, NeuralNetworkTrainer
from collections import deque
from Snake_AI import SnakeGameAI, Direction, Point


learning_rate = 0.001
memory_capacity = 100_000
training_batch_size = 1000

def training():
    plot_scores = []
    plot_avg_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # training short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # training long memory, plot result
            game.reset()
            agent.games_played += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save_model()

            print('Game', agent.games_played, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.games_played
            plot_avg_score.append(mean_score)
            plot(plot_scores, plot_avg_score)


class Agent:


    def __init__(self):
        self.discount_factor = 0.9 
        self.games_played = 0
        self.exploration_rate = 0 
        self.model = DeepQNetwork(11, 256, 3)
        self.memory = deque(maxlen=memory_capacity) 
        self.trainer = NeuralNetworkTrainer(self.model, lr=learning_rate, discount_factor=self.discount_factor)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if memory_capacity is reached

    def train_long_memory(self):
        if len(self.memory) > training_batch_size:
            mini_sample = random.sample(self.memory, training_batch_size) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.exploration_rate = 80 - self.games_played
        final_move = [0,0,0]
        if random.randint(0, 200) < self.exploration_rate:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


if __name__ == '__main__':
    training()
