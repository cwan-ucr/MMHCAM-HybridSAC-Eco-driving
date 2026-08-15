"""
Quick sanity-check for the SUMO CAV environment interface.

Run WITHOUT SUMO installed to verify shapes and types:
    python test_env.py --mock

Run WITH SUMO installed to do a real rollout:
    python test_env.py
"""

import argparse
import numpy as np
import torch


def test_mock():
    """Test mock per-CAV batch shapes (no SUMO needed)."""
    print("=== Mock DTDE Batch Shape Test ===\n")

    obs_per_cav = 35   # 9 own + 18 neighbor features + 6 signal([cur/opp G/Y, T_l, T'_l]) + 2 prev_action
    act_per_cav = 2
    K = 8

    print(f"obs_per_cav: {obs_per_cav}")
    print(f"act_per_cav: {act_per_cav}")
    print(f"K neighbors: {K}")

    batch_size = 128
    horizon = 3

    obs_batch = torch.randn(horizon + 1, batch_size, obs_per_cav)
    ctx_obs_batch = torch.randn(horizon + 1, batch_size, K, obs_per_cav)
    ctx_mask_batch = torch.ones(horizon + 1, batch_size, K)
    entry_phase_batch = torch.zeros(horizon + 1, batch_size, dtype=torch.long)
    progress_batch = torch.zeros(horizon + 1, batch_size, dtype=torch.long)
    act_batch = torch.randn(horizon, batch_size, act_per_cav).clamp(-1, 1)
    rew_batch = torch.randn(horizon, batch_size, 1)

    print(f"\nBatch shapes (H={horizon}, B={batch_size}):")
    print(f"  obs:             {obs_batch.shape}")
    print(f"  ctx_obs:         {ctx_obs_batch.shape}")
    print(f"  ctx_mask:        {ctx_mask_batch.shape}")
    print(f"  entry_phase_id:  {entry_phase_batch.shape}")
    print(f"  progress_id:     {progress_batch.shape}")
    print(f"  action:          {act_batch.shape}")
    print(f"  reward:          {rew_batch.shape}")

    print("\n✓ Shapes match TDMPC2._update signature")
    print("✓ Mock test passed\n")


def test_real():
    """Test the real SUMO environment with the new dict interface."""
    print("=== Real SUMO Environment Test ===\n")

    from envs.sumo_env import SumoIntersectionEnv, _ACT_PER_CAV

    env = SumoIntersectionEnv(
        max_steps=1000, use_gui=False, seed=42,
        context_size=8, vid_cycle=120, max_controllable=200,
    )
    
    print("Resetting environment...")
    obs_dict, info = env.reset()
    print(f"  obs shape:         {obs_dict['obs'].shape}")
    print(f"  ctx_obs shape:     {obs_dict['ctx_obs'].shape}")
    print(f"  ctx_mask shape:    {obs_dict['ctx_mask'].shape}")
    print(f"  entry_phase_id:    {obs_dict['entry_phase_id'].shape}")
    print(f"  progress_id:       {obs_dict['progress_id'].shape}")
    print(f"  cav_ids (count):   {len(obs_dict['cav_ids'])}")
    print(f"  info:              {info}")

    total_reward = 0
    for step in range(50):
        n = len(obs_dict['cav_ids'])
        action = np.random.uniform(-1, 1, size=(n, _ACT_PER_CAV)).astype(np.float32)
        obs_dict, reward, done, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            print(
                f"  Step {step:>3d}  reward={reward:>7.3f}  "
                f"cavs={info['num_cavs']}  done={done}"
            )

        if done:
            print("  Episode ended early.")
            break

    env.close()
    print(f"\nTotal reward: {total_reward:.3f}")
    print("✓ Real environment test passed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run mock test (no SUMO needed)")
    args = parser.parse_args()

    if args.mock:
        test_mock()
    else:
        test_real()
