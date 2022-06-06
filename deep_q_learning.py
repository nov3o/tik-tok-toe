import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
import random


class TicTacNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl1 = nn.Linear(9, 128)
        self.dl2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 9)

    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

class NetContext:
    def __init__(self, policy_net, target_net, optimizer, loss_function):
        self.policy_net = policy_net

        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optimizer
        self.loss_function = loss_function
        
class Player:
    
    def __init__(self, learning_rate, discount_factor, exp_rate):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.exp_rate = exp_rate
        self.env = TictactoeEnv()
        
    def play_training_games(self, net_context, discount_factor, N, epsilon = 0., print_every = 100):
        
        Turns = np.array(['X','O'])
        avg_reward = 0
        for game in range(N):
            #print("New game :", game)
            self.env.reset()
            grid, _, __ = self.env.observe()
            
            #Turns = Turns[::-1]
            
            player_opt = OptimalPlayer(epsilon, player=Turns[0])
            
            move_history = deque()
            reward = -9
            #move_history, reward = play_game(epsilon, print_every, player_opt, Turns[1])
            #print("PLAYER IS :", Turns[1])
            for j in range(9):
                if self.env.current_player == player_opt.player:
                    move_tuple = player_opt.act(grid)
                    move = 3*move_tuple[0] + move_tuple[1]
                    grid, end, winner = self.env.step(move, print_grid=False)
                    #print("MOVE autre :", move)
                    
                else:
                    #print("I move")
                    move = self.act(grid, Turns[1], net_context.target_net)
                    
                    if grid.reshape(9)[move] != 0:
                        #print("can't do the move")
                        move_history.appendleft([grid.reshape(9), move])
                        reward = -1
                        self.env.reset()
                        #print("Move history no move :", move_history)
                        avg_reward += reward
                        break
                        
                    else:
                        move_history.appendleft([grid.reshape(9), move])
                        grid, end, winner = self.env.step(move, print_grid=False)
                        #print("MOVE moi :", move)
                if end:
                    #print("Player :", Turns[1], "Winner :", winner)
                    reward = self.env.reward(Turns[1])
                    #print("Move history :", move_history)
                    self.env.reset()
                    avg_reward += reward
                    break
            if game%250 == 0:
                            print("Game :", game, "AVERGAE REWARD :", avg_reward/250)
                            avg_reward = 0
                    
            #print("REWARD :", reward)
            self.update_training_gameover(net_context, move_history, discount_factor, reward)

            if (game+1) % (print_every) == 0:
                #epsilon = max(0, epsilon - 0.1)
                print(f"{game+1}/{N} games, using epsilon={epsilon}...")
            
            
    def update_training_gameover(self, net_context, move_history, discount_factor, reward):
        
        game_result_reward = reward

        # move history is in reverse-chronological order - last to first
        next_position, move_index = move_history[0]
        print("NEXT :", next_position, move_index, reward)

        self.backpropagate(net_context, next_position, move_index, game_result_reward)

        for (position, move_index) in list(move_history)[1:]:
            with torch.no_grad():
                next_q_values = self.get_q_values(next_position, net_context.target_net)
                qv = torch.max(next_q_values).item()

            self.backpropagate(net_context, position, move_index, self.discount_factor * qv)

            next_position = position

        net_context.target_net.load_state_dict(net_context.policy_net.state_dict())
   

    def backpropagate(self, net_context, position, move_index, target_value):
        net_context.optimizer.zero_grad()
        #print("position :", position)
        output = net_context.policy_net(torch.tensor(position, dtype=torch.float))

        target = output.clone().detach()
        target[move_index] = target_value
        illegal_move_indexes = self.get_illegal_move_indexes(position)
        for mi in illegal_move_indexes:
            target[mi] = -1
        
        #print("ILLEGAL MOVES :", illegal_move_indexes)
        #print("POSITION", position)
        #print("move_index :", move_index)
        #print("OUTPUT :", output)
        #print("TARGET :", target)
        loss = net_context.loss_function(output, target)
        #print("LOSS :", loss)
        #print("loss backward :", loss.gradient)
        net_context.optimizer.step()
        

    def act(self, grid, symbol, model):
            if np.random.uniform(0, 1) <= self.exp_rate:
                action_index = self.randomMove(grid)
                #action_index = 3*action[0] + action[1]
                #print("random")
                
            else:
                #positions = self.availablePositions(grid)
                with torch.no_grad():
                    q_values = self.get_q_values(grid, model)
                    #print("Q_values :", q_values)
                    action_index = torch.argmax(q_values).item()

            return action_index

    def get_q_values(self, grid, model):
        inputs = torch.tensor(grid.reshape(9), dtype=torch.float)
        #print("input :", inputs)
        outputs = model(inputs)
        #print("output :", outputs)
        return outputs
    
    def get_illegal_move_indexes(self, position):
        return ([i for i in range(position.size) if position.reshape(9)[i] != 0])
    
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
                avail.append(3*pos[0] + pos[1])
        return avail
    
    def test_policy(self, N_test, epsilon = 0.):
        Turns = np.array(['X','O'])
        n_wins = 0
        n_loss = 0
        self.exp_rate = -1
        
        for i in range(N_test):
            self.env.reset()
            grid, _, __ = self.env.observe()
            #Turns = Turns[::-1]
            player_opt = OptimalPlayer(epsilon, player=Turns[0])
            print("Player :", Turns[1])
            for j in range(9):
                if self.env.current_player == player_opt.player:
                    move_tuple = player_opt.act(grid)
                    move = 3*move_tuple[0] + move_tuple[1]
                    grid, end, winner = self.env.step(move, print_grid=True)
                    
                else:
                    move = self.act(grid, Turns[1], net_context.target_net)
                    
                    if grid.reshape(9)[move] != 0:
                        print("can't do the move")
                        n_loss += 1
                        self.env.reset()
                        break
                        
                    else:
                        grid, end, winner = self.env.step(move, print_grid=True)
            
                if end:
                    if winner == Turns[1]:
                        n_wins +=1
                        print("n_wins", n_wins)
                    if winner == Turns[0]:
                        n_loss +=1
                        print("n_loss :", n_loss)
                    self.env.reset()
                    break
                    
        return (n_wins-n_loss)/N_test


import numpy as np

import torch
from torch.nn import MSELoss, HuberLoss

learning_rate = 0.05
discount_factor = 0.99
exp_rate = 0.8
optimal_eps_train = 0.5

my_player = Player(learning_rate, discount_factor, exp_rate)

policy_net = TicTacNet()
target_net = TicTacNet()
adam = torch.optim.Adam(policy_net.parameters(), lr= learning_rate)
loss = HuberLoss(delta = 1)

net_context = NetContext(policy_net, target_net, adam, loss)

print("Training qlearning X vs. random...")
my_player.play_training_games(net_context = net_context, discount_factor = 0.99, N= 20000, epsilon = optimal_eps_train, print_every = 1000)

#print("Training qlearning O vs. random...")
#play_training_games_o(net_context=net_context, total_games= 10000,
#                      x_strategies=[play_random_move])
print("")


a = my_player.test_policy(500, epsilon = 0.5)
