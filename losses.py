import time
import ray
import torch
import numpy as np
from model import ActorCritic, masked_softmax_logits
from worker import RolloutWorker


def compute_returns(rewards, gamma):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return np.array(list(reversed(out)), dtype=np.float32)


def init_ray(ray_address=None):
    """
    If ray_address is provided, try to connect to an existing Ray cluster.
    If connection fails or no address provided, start local Ray.
    """
    if ray.is_initialized():
        return

    if ray_address:
        try:
            ray.init(address=ray_address, ignore_reinit_error=True)
            return
        except Exception:
            pass

    ray.init(ignore_reinit_error=True)


def train(
    num_workers=4,
    iters=200,
    episodes_per_worker=50,
    lr=3e-4,
    gamma=0.99,
    vf_coef=0.5,
    ent_coef=0.01,
    batch_max_steps=5000,
    ray_address=None,
):
    init_ray(ray_address)

    total_start = time.perf_counter()

    device = "cpu"
    model = ActorCritic().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    workers = [RolloutWorker.remote(seed=i) for i in range(num_workers)]

    def broadcast():
        weights = model.state_dict()
        ray.get([wk.set_weights.remote(weights) for wk in workers])

    broadcast()

    for it in range(1, iters + 1):
        t0 = time.time()

        results = ray.get(
            [wk.run_episodes.remote(episodes_per_worker, gamma) for wk in workers]
        )

        trajs = []
        agg = {"wins": 0, "losses": 0, "draws": 0, "illegal": 0, "episodes": 0}
        placements = []

        for t, st, info in results:
            trajs += t
            for k in agg:
                agg[k] += st[k]
            placements.append(info)

        placement_str = ", ".join(
            f"{p['seed']}@{p['host']}({p['ip']})"
            for p in sorted(placements, key=lambda x: x["seed"])
        )
        print(f"[it {it:04d}] rollout_nodes: {placement_str}")

        obs_list, act_list, ret_list = [], [], []
        for tr in trajs:
            rets = compute_returns(tr["rew"], gamma)
            obs_list.append(tr["obs"])
            act_list.append(tr["act"])
            ret_list.append(rets)

        obs = torch.tensor(np.concatenate(obs_list), dtype=torch.float32, device=device)
        act = torch.tensor(np.concatenate(act_list), dtype=torch.int64, device=device)
        ret = torch.tensor(np.concatenate(ret_list), dtype=torch.float32, device=device)

        if obs.shape[0] > batch_max_steps:
            idx = torch.randperm(obs.shape[0])[:batch_max_steps]
            obs, act, ret = obs[idx], act[idx], ret[idx]

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

    total_time = time.perf_counter() - total_start
    print(f"\nTotal training time: {total_time:.2f} seconds")
    print("Training complete")

    return model
