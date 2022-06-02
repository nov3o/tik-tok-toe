import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
import random

class Player:
    def __init__(self, learning_rate, discount_factor, exp_rate):
        self.rand = 1
        self.states = []
        self.states_value = {}  # state -> value
        self.lr = learning_rate
        self.decay_gamma = discount_factor
        self.exp_rate = exp_rate
        self.env = TictactoeEnv()
        
    #def train(self, N, eps):
        
    def act(self, grid, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = self.randomMove(grid)
        else:
            positions = self.availablePositions(grid)
            value_max = -999
            if symbol == 'X':
                symb = 1
            elif symbol == 'O':
                symb = -1
            else:
                print("ERROR: wrong symbol")
            for p in positions:
                next_board = grid.copy()
                next_board[p] = symb
                value = 0 if self.states_value.get( str(next_board.reshape(9))) is None else self.states_value.get(str(next_board.reshape(9)))
                if value >= value_max:
                    value_max = value
                    action = p
            # print("{} takes action {}".format(self.name, action))
        return action
    
    def randomMove(self, grid):
        """ Chose a random move from the available options. """
        avail = self.availablePositions(grid)

        return avail[random.randint(0, len(avail)-1)]
         
    def availablePositions(self, grid): #nommé empty dans env
        '''return all empty positions'''
        avail = []
        for i in range(9):
            pos = (int(i/3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail
    
    
    def update_qtable(self, reward):
        for state in reversed(self.states):
            if self.states_value.get(state) is None:
                self.states_value[state] = 0
            self.states_value[state] += self.lr*(self.decay_gamma*reward - self.states_value[state])
            reward = self.states_value[state]
            
    def addState(self, state):
        self.states.append(str(state))
            
    
    def train(self, N, epsilon = 0., print_every = 100):
        Turns = np.array(['X','O'])
        avg_reward = 0
        for i in range(N):
            self.exp = i
            self.env.reset()
            grid, _, __ = self.env.observe()
            Turns = Turns[::-1]
            player_opt = OptimalPlayer(epsilon, player=Turns[0])
                
            for j in range(9):
                if self.env.current_player == player_opt.player:
                    move = player_opt.act(grid)
                else:
                    move = self.act(grid, Turns[1])

                grid, end, winner = self.env.step(move, print_grid=False)
                self.addState(self.env.grid.reshape(9))
            
                if end:
                    if i%print_every == 0:
                        print("Game n°:", i, "exp :", self.exp)
                        print('-------------------------------------------')
                        print('Game end, winner is player ' + str(winner))
                        print('Optimal player = ' +  Turns[0])
                        print('Player = ' +  Turns[1])
                        self.env.render()
                        print("AVERGAE REWARD :", avg_reward/print_every)
                        avg_reward = 0
                    reward = self.env.reward(Turns[1])
                    avg_reward += reward
                    self.update_qtable(reward)
                    self.env.reset()
                    break
                    
    def test_policy(self, N_test, epsilon = 0.):
        Turns = np.array(['X','O'])
        n_wins = 0
        n_loss = 0
        self.rand = 0
        
        for i in range(N_test):
            self.env.reset()
            grid, _, __ = self.env.observe()
            Turns = Turns[::-1]
            player_opt = OptimalPlayer(epsilon, player=Turns[0])
                
            for j in range(9):
                if self.env.current_player == player_opt.player:
                    move = player_opt.act(grid)
                else:
                    move = self.act(grid, Turns[1] )

                grid, end, winner = self.env.step(move, print_grid=False)
            
                if end:
                    if winner == Turns[1]:
                        n_wins +=1
                    if winner == Turns[0]:
                        n_loss +=1
                    self.env.reset()
                    break
                    
        return (n_wins-n_loss)/N_test

#TRAIN PARAMETERS
learning_rate = 0.05
discount_factor = 0.99
exp_rate = 0.5
optimal_eps_train = 1.

my_player = Player(learning_rate, discount_factor, lim_exp)

#TRAIN
games_to_train = 6000
print_every = 250
my_player.train(games_to_train, optimal_eps_train, print_every)

#TEST
optimal_eps_test = 0.5
games_to_test = 500
M = my_player.test_policy(games_to_test, optimal_eps_test)
print(M)
