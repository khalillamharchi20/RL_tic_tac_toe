import random
import numpy as np

class TicTacToeEnv:
    def __init__(self, opponent="random"):
        self.opponent = opponent
        self.reset()

    def reset(self):
        self.state = [0] * 9
        self.done = False
        self.winner = 0
        return self._obs()

    def _obs(self):
        # observation is always from the learning agent view
        return np.array(self.state, dtype=np.float32)

    def get_available_moves(self, state=None):
        if state is None:
            state = self.state
        return [i for i in range(9) if state[i] == 0]

    def check_game_over(self):
        s = self.state
        lines = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        for a,b,c in lines:
            if s[a] == s[b] == s[c] != 0:
                return True, s[a]
        if 0 not in s:
            return True, 0
        return False, 0

    def _opponent_move(self):
        moves = self.get_available_moves()
        if not moves:
            return
        if self.opponent == "random":
            a = random.choice(moves)
        else:
            a = random.choice(moves)
        self.state[a] = -1  # opponent is -1

    def step(self, action):
        if self.done:
            return self._obs(), 0.0, True, {}

        # illegal move
        if self.state[action] != 0:
            self.done = True
            self.winner = -1
            return self._obs(), -1.0, True, {"illegal": True}

        # agent plays +1
        self.state[action] = 1
        self.done, self.winner = self.check_game_over()
        if self.done:
            r = 1.0 if self.winner == 1 else 0.0
            return self._obs(), r, True, {}

        # opponent plays -1
        self._opponent_move()
        self.done, self.winner = self.check_game_over()
        if self.done:
            r = -1.0 if self.winner == -1 else 0.0
            return self._obs(), r, True, {}

        return self._obs(), 0.0, False, {}
    
    def render(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("\n")
        for i in range(3):
            row = [symbols[self.state[i*3 + j]] for j in range(3)]
            print(" " + " | ".join(row))
            if i < 2:
                print("---+---+---")
        print("\n")