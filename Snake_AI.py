import random
import numpy as np
import pygame
from collections import namedtuple
from enum import Enum


pygame.init()
font = pygame.font.Font('arial.ttf', 30)

block_size = 20
game_speed = 40


Point = namedtuple('Point', 'x, y')

# rgb colors
white_color = (255, 255, 255)
red_color = (255,0,0)
green_color = (0, 255, 0)
dark_green_color = (0, 170, 0)
black_color = (0,0,0)



class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Snake')
        self.reset()


    def _place_food(self):
        x = random.randint(0, (self.w-block_size )//block_size )*block_size
        y = random.randint(0, (self.h-block_size )//block_size )*block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-block_size, self.head.y),
                      Point(self.head.x-(2*block_size), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - block_size or pt.x < 0 or pt.y > self.h - block_size or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def play_step(self, action):
        self.frame_iteration += 1
        #collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        #move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        #check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        #place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        #update ui and clock
        self._update_ui()
        self.clock.tick(game_speed)
        #return game over and score
        return reward, game_over, self.score




    def _update_ui(self):
        self.display.fill(black_color)

        for pt in self.snake:
            pygame.draw.rect(self.display, green_color, pygame.Rect(pt.x, pt.y, block_size, block_size))
            pygame.draw.rect(self.display, dark_green_color, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, red_color, pygame.Rect(self.food.x, self.food.y, block_size, block_size))

        text = font.render("Score: " + str(self.score), True, white_color)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += block_size
        elif self.direction == Direction.LEFT:
            x -= block_size
        elif self.direction == Direction.DOWN:
            y += block_size
        elif self.direction == Direction.UP:
            y -= block_size

        self.head = Point(x, y)