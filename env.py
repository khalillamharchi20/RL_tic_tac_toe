import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class TicTacToeEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # State as a list of 9 numbers: 0=empty, 1=player, -1=opponent
        self.state = [0] * 9
        self.current_player = 1  
        self.done = False
        self.winner = 0
        return self.state.copy()
    
    def get_available_moves(self, state=None):
        if state is None:
            state = self.state
        return [i for i in range(9) if state[i] == 0]
    
    def step(self, action):

        if self.state[action] != 0:
            return self.state, -10, True, {}  
        

        self.state[action] = self.current_player
        

        self.done, self.winner = self.check_game_over()
        

        if self.done:
            if self.winner == 1:
                reward = 1
            elif self.winner == -1:
                reward = -1
            else:
                reward = 0
        else:
            reward = 0
            self.current_player *= -1
        
        return self.state.copy(), reward, self.done, {}
    
    def check_game_over(self):
        state = self.state
        
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  
            [0, 4, 8], [2, 4, 6]              
        ]
        
        for line in lines:
            if state[line[0]] == state[line[1]] == state[line[2]] != 0:
                return True, state[line[0]]
        
        if 0 not in state:
            return True, 0
        
        return False, 0
    
    def render(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("\n")
        for i in range(3):
            row = [symbols[self.state[i*3 + j]] for j in range(3)]
            print(" " + " | ".join(row))
            if i < 2:
                print("---+---+---")
        print("\n")