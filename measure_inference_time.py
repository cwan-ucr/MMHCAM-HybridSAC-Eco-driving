from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace

import torch

from agents.sac import SAC


def make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        obs_per_agent=35,
        act_per_agent=2,
        lane_action_discrete=True,
        sac_hidden_dim=128,
        mlp_dim=128,
        sac_feature_dim=128,
        latent_dim=128,
        attention_heads=4,
        attention=True,
        communication=True,
        sac_attention_include_self_token=True,
        attention_fusion_gate=False,
        sac_input_dropout=0.0,
        sac_log_freq=50,
        lr=3e-4,
        episode_length=1500,
        discount_denom=5,
        discount_min=0.95,
        discount_max=0.99,
        tau=0.005,
        grad_clip_norm=20,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def time_policy(agent: SAC, batch_size: int, warmup: int, repeats: int) -> dict[str, float]:
    device = agent.device
    own = torch.randn(batch_size, agent.cfg.obs_per_agent, device=device)
    ctx = torch.randn(batch_size, 8, agent.cfg.obs_per_agent, device=device)
    mask = torch.ones(batch_size, 8, device=device)
    obs = {
        "cav_ids": list(range(batch_size)),
        "obs": own,
        "ctx_obs": ctx,
        "ctx_mask": mask,
    }

    for _ in range(warmup):
        _ = agent.act(obs, eval_mode=True)
    synchronize(device)

    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = agent.act(obs, eval_mode=True)
        synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": sorted(times_ms)[int(0.95 * (len(times_ms) - 1))],
        "per_cav_mean_us": statistics.fmean(times_ms) * 1000.0 / batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/sumo-intersection/1/cav_control/models/final.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()

    cfg = make_cfg()
    agent = SAC(cfg)
    agent.load(args.checkpoint)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    agent.device = device
    agent.to(device)
    agent.eval()

    print(f"device,batch_size,mean_ms,median_ms,p95_ms,per_cav_mean_us")
    for batch_size in args.batches:
        stats = time_policy(agent, batch_size, args.warmup, args.repeats)
        print(
            f"{device.type},{batch_size},"
            f"{stats['mean_ms']:.6f},{stats['median_ms']:.6f},"
            f"{stats['p95_ms']:.6f},{stats['per_cav_mean_us']:.6f}"
        )


if __name__ == "__main__":
    main()
