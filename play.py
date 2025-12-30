import torch
import numpy as np
from env import TicTacToeEnv
from model import ActorCritic, masked_softmax_logits

def load_model(path='saved_models/model.pt'):
    """Load trained model"""
    model = ActorCritic()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def get_model_move(model, state):
    """Get model's move for a given state"""
    obs_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs_t)
        logits = masked_softmax_logits(logits, obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
    return action

def human_move(state):
    """Get human player's move"""
    while True:
        try:
            move = int(input("Enter your move (0-8, row-major): "))
            if move < 0 or move > 8:
                print("Move must be between 0 and 8")
                continue
            if state[move] != 0:
                print("That cell is already occupied!")
                continue
            return move
        except ValueError:
            print("Please enter a valid number")

def display_board(state):
    """Display the board with position numbers"""
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print("\nBoard positions:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")
    
    print("\nCurrent board:")
    for i in range(3):
        row = [symbols[state[i*3 + j]] for j in range(3)]
        print(" " + " | ".join(row))
        if i < 2:
            print("---+---+---")
    print()

def play_game(model, human_first=True):
    """Play a game against the trained model using step_single"""
    env = TicTacToeEnv(opponent="random")
    state = env.reset()
    
    print("\n" + "="*40)
    print("NEW GAME")
    print("You are 'X', the AI is 'O'" if human_first else "You are 'O', the AI is 'X'")
    print("="*40)
    
    display_board(state)
    
    current_player = 1 if human_first else -1  # 1 = X, -1 = O
    
    while not env.done:
        if current_player == 1:  # X's turn
            if human_first:
                # Human plays X
                move = human_move(state)
                state, _, done, _ = env.step_single(move, 1)
            else:
                # AI plays X
                move = get_model_move(model, state)
                print(f"AI (X) chooses position {move}")
                state, _, done, _ = env.step_single(move, 1)
        else:  # O's turn
            if not human_first:
                # Human plays O
                move = human_move(state)
                state, _, done, _ = env.step_single(move, -1)
            else:
                # AI plays O
                move = get_model_move(model, state)
                print(f"AI (O) chooses position {move}")
                state, _, done, _ = env.step_single(move, -1)
        
        display_board(state)
        
        # Switch player
        current_player = -current_player
    
    # Game result
    if env.winner == 1:
        if human_first:
            print(" You win! (X wins)")
        else:
            print(" AI wins! (X wins)")
    elif env.winner == -1:
        if not human_first:
            print(" You win! (O wins)")
        else:
            print(" AI wins! (O wins)")
    else:
        print(" It's a draw!")
    
    return env.winner

def main():
    # Load trained model
    try:
        model = load_model('saved_models/model.pt')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("No trained model found. Please train the model first.")
        print("Run: python main.py or python export_model.py")
        return
    
    print("\n" + "="*40)
    print("TIC-TAC-TOE AI PLAYER")
    print("="*40)
    
    scores = {"human": 0, "ai": 0, "draws": 0}
    
    while True:
        print("\nCurrent score:")
        print(f"  You: {scores['human']}")
        print(f"  AI: {scores['ai']}")
        print(f"  Draws: {scores['draws']}")
        print("\n1. Play as X (first)")
        print("2. Play as O (second)")
        print("3. Reset scores")
        print("4. Quit")
        
        choice = input("\nSelect option (1-5): ")
        
        if choice == '1':
            winner = play_game(model, human_first=True)
            if winner == 1:
                scores["human"] += 1
            elif winner == -1:
                scores["ai"] += 1
            else:
                scores["draws"] += 1
                
        elif choice == '2':
            winner = play_game(model, human_first=False)
            if winner == -1:
                scores["human"] += 1
            elif winner == 1:
                scores["ai"] += 1
            else:
                scores["draws"] += 1
                
              
        elif choice == '3':
            scores = {"human": 0, "ai": 0, "draws": 0}
            print("Scores reset!")
            
        elif choice == '4':
            print("\nFinal score:")
            print(f"  You: {scores['human']}")
            print(f"  AI: {scores['ai']}")
            print(f"  Draws: {scores['draws']}")
            print("\nThanks for playing!")
            break
            
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()