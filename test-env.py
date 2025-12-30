import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import TicTacToeEnv


def simple_test():

    env = TicTacToeEnv()
    
    print("Test 1: Game NOT done")
    env.state = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    done, winner = env.check_game_over()
    print(f"  Board: {env.state}")
    print(f"  Done: {done} (expected: False)")
    print(f"  Winner: {winner}")
    
    print("\nTest 2: Game IS done")
    env.state = [1, 1, 1, 0, -1, -1, 0, 0, 0]
    done, winner = env.check_game_over()
    print(f"  Board: {env.state}")
    print(f"  Done: {done} (expected: True)")
    print(f"  Winner: {winner} (expected: 1)")


if __name__ == "__main__":
    simple_test()