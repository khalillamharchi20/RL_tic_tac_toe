import ray
import numpy as np
import torch
from env import TicTacToeEnv
from model import ActorCritic, masked_softmax_logits


@ray.remote(num_cpus=1)
class RolloutWorker:
    def __init__(self, seed=0):
        import random
        random.seed(seed)
        np.random.seed(seed)

        self.env = TicTacToeEnv(opponent="random")
        self.device = "cpu"
        self.model = ActorCritic().to(self.device)
        self.model.eval()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def run_episodes(self, n_episodes=50, gamma=0.99):
        trajectories = []
        stats = {"wins": 0, "losses": 0, "draws": 0, "illegal": 0, "episodes": 0}

        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False

            ep_obs, ep_act, ep_rew, ep_val, ep_logp = [], [], [], [], []

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, value = self.model(obs_t)
                    logits = masked_softmax_logits(logits, obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
                    logp = dist.log_prob(torch.tensor(action)).item()
                    v = value.item()

                next_obs, r, done, info = self.env.step(action)

                ep_obs.append(obs)
                ep_act.append(action)
                ep_rew.append(r)
                ep_val.append(v)
                ep_logp.append(logp)

                obs = next_obs

                if done and info.get("illegal"):
                    stats["illegal"] += 1

            if self.env.winner == 1:
                stats["wins"] += 1
            elif self.env.winner == -1:
                stats["losses"] += 1
            else:
                stats["draws"] += 1
            stats["episodes"] += 1

            trajectories.append(
                {
                    "obs": np.array(ep_obs, dtype=np.float32),
                    "act": np.array(ep_act, dtype=np.int64),
                    "rew": np.array(ep_rew, dtype=np.float32),
                    "val": np.array(ep_val, dtype=np.float32),
                    "logp": np.array(ep_logp, dtype=np.float32),
                }
            )

        return trajectories, stats
