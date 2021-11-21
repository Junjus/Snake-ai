# Learn and Progress Steps in the Project

## 1. Phase: Setting up git repository & snake script

## 2.Phase: Getting started with pytorch

1. Watched a [youtube tutorial](https://www.youtube.com/watch?v=c36lUUr864M) about pytorch.
    1. Installing Cuda for GPU-processing
    Installing pytorch for machine-learning

    2. Basic tensor-operations

    3. Autograd for gradiants for weight optimization

    4. Backpropagation to adjust weights by using the gradiants

## Snake Game

### Modify playground

1. reset
The snake game will be reset if the snake dies.
before: you started it and then it initilized itself and if the game was over it killed it self.
now once you game over it will restart.

2. reward
this is needed to tell the ai the quality of its moves.
move = 0
food = +10
game_over = -10

3. human input changed to agent input
removed use of keyinputs an inserted action from agent
agent inputs are 3 insed of 4. --> easier for ai
insted of left, right, up, down its turn left, turn right, none

4. added game iteration function to keep track of interations, and if the snake moves withur doing anything for too long.

5. modified is_boundary function
to allow tracking the distance to the obsticals.
