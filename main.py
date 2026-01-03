import argparse
from losses import train
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ray-address", default="", help='"" for local, "auto" or "IP:6379" for cluster')
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--episodes-per-worker", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--batch-max-steps", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ray_address = args.ray_address.strip() or None

    _ = train(
        num_workers=args.num_workers,
        iters=args.iters,
        episodes_per_worker=args.episodes_per_worker,
        lr=args.lr,
        gamma=args.gamma,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        batch_max_steps=args.batch_max_steps,
        ray_address=ray_address,
    )
    print("Training complete")
    torch.save(model.state_dict(), 'saved_models/model.pt')
