from math import gamma, nan
from typing import List
from numpy.lib.type_check import nan_to_num
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

from torch._C import ParameterDict
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discovery rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


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
        self.memory.append((state, action, reward, next_state, done)) # pop left if MAX_MEMORY is exeded

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move, move

# calculates the correlation between moves and inputs
def calcCorrelation(correlationState, anotherArg = 0):

    correlationState = np.asarray(correlationState, np.int8)

    
    # Shows the direction the snake moves to
    #
    # moveDirsV:   -1:Down 0: None 1:Up
    # moveDirsH:   -1:Left 0: None 1:Right
    # moveDirs:     0:Up 1:Right 2:Down 3:Left
    moveDirs = []
    moveDirsH = []
    moveDirsV = []
    for idx in range(len(correlationState[0])):

        if correlationState[4][idx] == 1:
            moveDir = 3
        if correlationState[5][idx] == 1:
            moveDir = 1
        if correlationState[6][idx] == 1:
            moveDir = 0
        if correlationState[7][idx] == 1:
            moveDir = 2

        if correlationState[0][idx] == 0:
            moveDirs.append(moveDir)                # no change
        elif correlationState[0][idx] == 1:
            moveDirs.append((moveDir + 1)%4)        # turn right       
        else:
            moveDirs.append((moveDir - 1)%4)        # turn left

    # Splites the moving direction into vertical and horizontal
    for move in moveDirs:
        if move == 0:   #Up
            moveDirsV.append(1)
            moveDirsH.append(0)
        elif move == 1: #Right
            moveDirsV.append(0)
            moveDirsH.append(1)
        elif move == 2: #Down
            moveDirsV.append(-1)
            moveDirsH.append(0)
        else:           #Left
            moveDirsV.append(0)
            moveDirsH.append(-1)
        
    # creates a unified list of the directions of dangers
    dangerDirs = []
    for idx in range(len(correlationState[0])):
        if correlationState[1][idx] == 1:
            dangerDirs.append(1)                    # danger streight
        elif correlationState[2][idx] == 1:
            dangerDirs.append(2)                    # danger right
        elif correlationState[3][idx] == 1:
            dangerDirs.append(3)                    # danger left
        else:
            dangerDirs.append(0)                    # no danger

    # creates a unified list of the direction of the food,
    # split in vertical and horzontal
    FoodsV = []
    FoodsH = []
    for idx in range(len(correlationState[0])):
        if (correlationState[8][idx] == 1) and (correlationState[9][idx] == 0) and (correlationState[10][idx] == 0) and (correlationState[11][idx] == 0):#Left & None
            FoodsH.append(0)
            FoodsV.append(-1)
        elif (correlationState[8][idx] == 1) and (correlationState[9][idx] == 0) and (correlationState[10][idx] == 1) and (correlationState[11][idx] == 0):#Left & Up
            FoodsH.append(1)
            FoodsV.append(-1)
        elif (correlationState[8][idx] == 0) and (correlationState[9][idx] == 0) and (correlationState[10][idx] == 1) and (correlationState[11][idx] == 0):#None & Up
            FoodsH.append(1)
            FoodsV.append(0)
        elif (correlationState[8][idx] == 0) and (correlationState[9][idx] == 1) and (correlationState[10][idx] == 1) and (correlationState[11][idx] == 0):#Right & Up
            FoodsH.append(1)
            FoodsV.append(1)
        elif (correlationState[8][idx] == 0) and (correlationState[9][idx] == 1) and (correlationState[10][idx] == 0) and (correlationState[11][idx] == 0):#Right & None
            FoodsH.append(0)
            FoodsV.append(1)
        elif (correlationState[8][idx] == 0) and (correlationState[9][idx] == 1) and (correlationState[10][idx] == 0) and (correlationState[11][idx] == 1):#Right & Down
            FoodsH.append(-1)
            FoodsV.append(1)
        elif (correlationState[8][idx] == 0) and (correlationState[9][idx] == 0) and (correlationState[10][idx] == 0) and (correlationState[11][idx] == 1):#None & Down
            FoodsH.append(-1)
            FoodsV.append(1)
        elif (correlationState[8][idx] == 1) and (correlationState[9][idx] == 0) and (correlationState[10][idx] == 0) and (correlationState[11][idx] == 1):#Left & Down
            FoodsH.append(-1)
            FoodsV.append(-1)

    return np.mean([nan_to_num(np.corrcoef(FoodsH, moveDirsH)[0,1]), nan_to_num(np.corrcoef(FoodsV, moveDirsV)[0,1])]), nan_to_num(np.corrcoef(dangerDirs , correlationState[0])[0,1])

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    correlationFood = []
    correlationDanger = []
    
    correlationState = [[],[],[],[],[],[],[],[],[],[],[],[]]

    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move, move = agent.get_action(state_old)

        for idx in range(12):
            if(idx ==0):
                correlationState[0].append(move)
            else:
                correlationState[idx].append(state_old[idx -1])

        # perform move and new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            # gets the values of the corralation
            corFood, corDan = calcCorrelation(correlationState)

            # adds the values of the correlation to the list to display the changes in correlation
            correlationFood.append(corFood)
            correlationDanger.append(corDan)

            # resets the list of the the moves used for calculating the correlations
            correlationState = [[],[],[],[],[],[],[],[],[],[],[],[]]

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores, correlationFood, correlationDanger)



if __name__ == '__main__':
    train()