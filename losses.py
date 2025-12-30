import time
import torch
import numpy as np
import ray
from model import ActorCritic, masked_softmax_logits
from worker import RolloutWorker

def compute_returns(rewards, gamma):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return np.array(list(reversed(out)), dtype=np.float32)

def train(num_workers=4, iters=200, episodes_per_worker=50, lr=3e-4, gamma=0.99,
          vf_coef=0.5, ent_coef=0.01, batch_max_steps=5000):

    ray.init(address="auto" if ray.is_initialized() else None, ignore_reinit_error=True)

    device = "cpu"
    model = ActorCritic().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    workers = [RolloutWorker.remote(seed=i) for i in range(num_workers)]

    def broadcast():
        w = model.state_dict()
        ray.get([wk.set_weights.remote(w) for wk in workers])

    broadcast()

    for it in range(1, iters+1):
        t0 = time.time()

        # gather rollouts
        results = ray.get([wk.run_episodes.remote(episodes_per_worker, gamma) for wk in workers])
        trajs = []
        agg = {"wins":0,"losses":0,"draws":0,"illegal":0,"episodes":0}
        for t, st in results:
            trajs += t
            for k in agg:
                agg[k] += st[k]

        # build a batch
        obs_list, act_list, ret_list = [], [], []
        for tr in trajs:
            rets = compute_returns(tr["rew"], tr["gamma"])
            obs_list.append(tr["obs"])
            act_list.append(tr["act"])
            ret_list.append(rets)

        obs = torch.tensor(np.concatenate(obs_list), dtype=torch.float32, device=device)
        act = torch.tensor(np.concatenate(act_list), dtype=torch.int64, device=device)
        ret = torch.tensor(np.concatenate(ret_list), dtype=torch.float32, device=device)

        # optional cap
        if obs.shape[0] > batch_max_steps:
            idx = torch.randperm(obs.shape[0])[:batch_max_steps]
            obs, act, ret = obs[idx], act[idx], ret[idx]

        # forward
        logits, val = model(obs)
        logits = masked_softmax_logits(logits, obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act)
        entropy = dist.entropy().mean()

        adv = (ret - val).detach()
        policy_loss = -(logp * adv).mean()
        value_loss = torch.mean((ret - val) ** 2)

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        broadcast()

        dt = time.time() - t0
        win_rate = agg["wins"] / max(1, agg["episodes"])
        print(
            f"[it {it:04d}] loss={loss.item():.4f} "
            f"pi={policy_loss.item():.4f} vf={value_loss.item():.4f} ent={entropy.item():.4f} "
            f"win={win_rate:.2%} (W/D/L={agg['wins']}/{agg['draws']}/{agg['losses']}) "
            f"illegal={agg['illegal']} steps={obs.shape[0]} time={dt:.2f}s"
        )

    return model
