import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.pi = nn.Linear(128, 9)
        self.v  = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value

def masked_softmax_logits(logits, obs):
    legal = (obs == 0)
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(legal, logits, torch.tensor(neg_inf, device=logits.device))
    return masked
