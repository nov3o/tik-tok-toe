import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
import matplotlib.pyplot as plt
import random

class Policy:
    def __init__(self, learning_rate, discount_factor, exp_rate, n_star):
        self.rand = 1
        self.states = []
        self.states_1 = []
        self.states_2 = []
        self.states_value = {}  # state -> value
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.exp_rate = exp_rate
        self.env = TictactoeEnv()
        self.player = 'X'
        self.exp_min = 0.1
        self.exp_max = 0.8
        self.n_star = n_star
        self.rewards_to_plot = []
        self.test_M_opt = []
        self.test_M_rand = []
        self.games_to_plot = []
        
    def act(self, grid, train):
        if train :
            exp = self.exp_rate
        else :
            exp = -1
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = self.randomMove(grid)
        else:
            positions = self.availablePositions(grid)
            value_max = -1000
            for p in positions:
                state_action = (grid.reshape(9), p, self.player)
                value = 0 if self.states_value.get( str(state_action)) is None else self.states_value.get(str(state_action))
                if value >= value_max:
                    value_max = value
                    action = p
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
    
    
    def update_qtable(self, reward, against_me = False):
        original_reward = reward
        if against_me == False:
            states = self.states
            for state in reversed(states):
                if self.states_value.get(state) is None:
                    self.states_value[state] = 0
                self.states_value[state] += self.lr*(self.discount_factor*reward - self.states_value[state])
                reward = self.states_value[state]
        else:
            states = self.states_1
            for i in range(2):
                for state in reversed(states):
                    if self.states_value.get(state) is None:
                        self.states_value[state] = 0
                    self.states_value[state] += self.lr*(self.discount_factor*reward - self.states_value[state])
                    reward = self.states_value[state]
                states = self.states_2
                reward = - original_reward
            
    def addState(self, state, action):
        self.states.append(str((state, action, self.player)))
        
          
    def train(self, N, epsilon = 0., print_every = 100):
        Turns = np.array(['X','O'])
        avg_reward = 0
        for i in range(N):
            self.env.reset()
            grid, _, __ = self.env.observe()
            Turns = Turns[::-1]
            player_opt = OptimalPlayer(epsilon, player=Turns[0])
            self.player = Turns[1]
            for j in range(9):
                if self.env.current_player == player_opt.player:
                    move = player_opt.act(grid)
                    grid, end, winner = self.env.step(move, print_grid=False)
                else:
                    position = grid
                    move = self.act(grid, train = True)
                    grid, end, winner = self.env.step(move, print_grid=False)
                    self.addState(position.reshape(9), move)
            
                if end:
                    if i%print_every == 0 and i!=0:
                        print("Game :", i, "AVERGAE REWARD :", avg_reward/print_every)
                        self.rewards_to_plot.append(avg_reward/250)
                        self.test_M_opt.append(self.test_policy(500, 0))
                        self.test_M_rand.append(self.test_policy(500, 1.))
                        #print("")
                        avg_reward = 0
                    reward = self.env.reward(Turns[1])
                    avg_reward += reward
                    self.update_qtable(reward)
                    self.states = []
                    self.env.reset()
                    break
                    
            self.exp_rate = max(self.exp_min, self.exp_max*(1 - i/self.n_star))
            if (i+1) % (1000) == 0:
                print(f"{i+1}/{N} games, using epsilon={self.exp_rate}...")
        #print("to_plot", len(self.rewards_to_plot))
        print()        
        #print("state_value :", self.states_value)
    
    
    def train_against_me(self, N, epsilon = 0., print_every = 100):
        Turns = np.array(['X','O'])
        avg_reward = 0
        for i in range(N):
            self.env.reset()
            grid, _, __ = self.env.observe()
            Turns = Turns[::-1]
            player_opt = OptimalPlayer(epsilon, player=Turns[0])
            self.player = Turns[1]
            for j in range(9):
                last_player = self.player
                if self.player == 'X':
                    self.player = 'O'
                    position = grid
                    move = self.act(grid, train = True)
                    grid, end, winner = self.env.step(move, print_grid=False)
                    self.addState(position.reshape(9), move)
                else:
                    self.player = 'X'
                    position = grid
                    move = self.act(grid, train = True)
                    grid, end, winner = self.env.step(move, print_grid=False)
                    self.addState(position.reshape(9), move)
            
                if end:
                    if i%print_every == 0 and i!=0:
                        print("Game :", i, "AVERGAE REWARD :", avg_reward/print_every)
                        self.rewards_to_plot.append(avg_reward/250)
                        self.test_M_opt.append(self.test_policy(500, 0))
                        self.test_M_rand.append(self.test_policy(500, 1.))
                        #print("")
                        avg_reward = 0
                    reward = self.env.reward(Turns[1])
                    avg_reward += reward
                    self.update_qtable(reward, against_me = True)
                    self.states = []
                    self.env.reset()
                    break
                    
            self.exp_rate = max(self.exp_min, self.exp_max*(1 - i/self.n_star))
            if (i+1) % (1000) == 0:
                print(f"{i+1}/{N} games, using epsilon={self.exp_rate}...")
        print()  
                    
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
            self.player = Turns[1]    
            for j in range(9):
                if self.env.current_player == player_opt.player:
                    move = player_opt.act(grid)
                else:
                    move = self.act(grid, train = False)

                grid, end, winner = self.env.step(move, print_grid=False)
            
                if end:
                    if winner == Turns[1]:
                        n_wins +=1
                    if winner == Turns[0]:
                        n_loss +=1
                    self.env.reset()
                    break
                    
        return (n_wins-n_loss)/N_test

learning_rate = 0.05
discount_factor = 0.99
exp_rate = 0.3
optimal_eps_train = 0.5
n_star = 10000

my_policy = Policy(learning_rate, discount_factor, exp_rate, n_star)

games_to_train = 20000
print_every = 250
my_policy.train(games_to_train, optimal_eps_train, print_every)

optimal_eps_test = 0.
games_to_test = 500
M = my_policy.test_policy(games_to_test, optimal_eps_test)
print(M)

games = range(250, 20000, 250)
plt.plot(games, my_policy.rewards_to_plot)

plt.plot(games, my_policy.test_M_opt)

plt.plot(games, my_policy.test_M_rand)

my_policy_against_me = Policy(learning_rate, discount_factor, exp_rate, n_star)

my_policy_against_me.train_against_me(games_to_train, optimal_eps_train, print_every)

plt.plot(games, my_policy_against_me.rewards_to_plot)

plt.plot(games, my_policy_against_me.test_M_rand)

plt.plot(games, my_policy_against_me.test_M_opt)
