#!/usr/bin/env python3
"""
Main entry point for A2C training on Tic-Tac-Toe.
Runs on a single node with Ray Core.
"""

from losses import train
import torch

if __name__ == "__main__":
    model = train(
        num_workers=4,
        iters=200,
        episodes_per_worker=50,
        lr=3e-4,
        gamma=0.99,
        vf_coef=0.5,
        ent_coef=0.01,
        batch_max_steps=5000
    )
    torch.save(model.state_dict(), 'saved_models/model.pt')
    print("Training complete!")
