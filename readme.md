# Tic-Tac-Toe AI with Reinforcement Learning

## Overview
This project implements a Reinforcement Learning (RL) agent that learns to play **Tic-Tac-Toe** using the **Advantage Actor-Critic (A2C)** algorithm.  
The agent is trained through **self-play against a random opponent** and can also be played interactively by a human user.

---

## Project Structure

### Core Components
- **env.py** – Tic-Tac-Toe game environment with an OpenAI Gym-like interface  
- **model.py** – Actor-Critic neural network with legal-move masking  
- **worker.py** – Ray remote workers for parallel episode collection  
- **losses.py** – A2C loss computation and training logic  
- **main.py** – Main training script  
- **play.py** – Interactive interface to play against the trained AI  

---

## Key Features
- ✅ **A2C Algorithm** – Stable policy gradient method  
- ✅ **Legal Move Masking** – Prevents illegal moves during training  
- ✅ **Parallel Training** – Distributed rollout collection using Ray  
- ✅ **Interactive Play** – Play against the trained AI  
- ✅ **Docker Support** – Easy, reproducible containerized setup  

---

## How It Works

### Training Process
1. Multiple Ray workers collect game episodes in parallel  
2. Workers play using the current policy against a random opponent  
3. Trajectories are collected (states, actions, rewards, values)  
4. A central trainer computes the A2C loss and updates the policy  
5. Updated weights are broadcast back to all workers  
6. The process repeats for a fixed number of iterations  

### Network Architecture
- **Input:** 9 values representing the board  
  - `1` = X, `-1` = O, `0` = empty  
- **Hidden Layers:**  
  - 2 fully connected layers with 128 units each  
- **Outputs:**  
  - **Policy head:** 9 action logits (with legal-move masking)  
  - **Value head:** Single scalar state value  

---

## Example: Training Output

Below is an example of running `main.py` to train the agent.  
It shows training iterations, loss values, win rate, and rollout statistics.

![Training Example](training_example.png)

---

## Quick Start

### Prerequisites
- Docker & Docker Compose (recommended)  
- Python 3.8+ (for local execution)  

---

## Running with Docker (Recommended)

### Build and start training
```bash
docker-compose -f docker-compose-local.yml up
```

This will:
- Build the Docker image  
- Train the model (e.g. 200 iterations with 4 parallel workers)  
- Save the trained model to `saved_models/model.pt`  
- Keep the container running for interactive play  

### Play against the trained AI
```bash
docker exec -it tictactoe_trainer python play.py
```

---

## Running Locally (Without Docker)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python main.py
```

### Play against the AI
```bash
python play.py
```

---

## Notes
- The agent quickly converges toward optimal play  
- Illegal moves are fully prevented by action masking  
- Parallel rollouts significantly speed up training  

Enjoy playing against your RL-powered Tic-Tac-Toe agent! 🎮🤖
