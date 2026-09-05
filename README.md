# Flexible Hybrid-Action CAV Eco-Driving

This repository contains the reproducible code for SUMO-based connected and automated vehicle (CAV) eco-driving experiments in mixed traffic. It includes the reinforcement-learning agents, SUMO environments, training and evaluation entry points, and scenario configuration files.

The repository is intentionally code-focused and does not include manuscript source files, raw evaluation outputs, FCD traces, or intermediate checkpoints. It includes the final checkpoints and episode-level training CSV files used for the paper experiments, together with representative result figures.

## Contents

```text
agents/                  RL agents: Hybrid SAC, PPO, DQN, TD-MPC2
common/                  Replay buffer, neural network layers, logging, metrics
envs/                    SUMO CAV intersection environments and wrappers
envs/sumo_files/         SUMO network, route, signal, and scenario files
trainer/                 Online/offline training loops
scripts/                 Convenience training and visualization scripts
tools/                   SUMO scenario construction utility
train.py                 Main training entry point
evaluate.py              RL policy evaluation entry point
evaluate_baseline.py     SUMO baseline evaluation
evaluate_glosa.py        SUMO GLOSA baseline evaluation
evaluate_comm_sensitivity.py  Communication-capacity evaluation
config.yaml              Hydra configuration
environment.yaml         Conda environment specification
figures/                 Representative training, evaluation, and trajectory results
artifacts/training_runs/ Final checkpoints and episode-level training CSV files
```

## Representative Results

The main comparison considers three learned control settings:

- `AV`: local perception only, without V2I signal timing or V2V CAV context.
- `CAV-V2I`: local perception plus signal timing information.
- `CAV-V2X`: local perception, signal timing, and neighboring-CAV communication.

Across five penetration rates and 100 matched evaluation runs per policy, the `CAV-V2X` policy gives the most balanced improvement among the learned policies. Averaged after run-level matching with the SUMO baseline, its mixed-traffic performance improves by `1.10%` in travel time, `56.21%` in stop time, `1.59%` in speed, `16.69%` in fuel consumption, `36.50%` in jerk, and `64.99%` in TET rate.

Algorithm training comparison:

![Algorithm training reward](figures/algorithm_training_reward.png)

![Algorithm training metrics](figures/algorithm_training_metrics.png)

Training reward comparison:

![Communication training reward](figures/communication_training_reward.png)

Training performance metrics:

![Communication training metrics](figures/communication_training_metrics.png)

Evaluation trajectories at 50% CAV penetration:

![50 percent CAV penetration trajectories](figures/trajectory_50pr_comparison.png)

Average savings by vehicle group:

![Mean savings polar plot](figures/mean_savings_polar.png)

Penetration sensitivity:

![Penetration sensitivity](figures/penetration_sensitivity.png)

Hybrid-action branch ablation:

![Hybrid-action ablation](figures/hybrid_action_ablation.png)

Available V2V-context sensitivity:

![Communication capacity sensitivity](figures/communication_capacity_heatmap.png)

The real-parameter Sycamore/Central scenario is used to test adaptation from a pretrained model. Pretraining improves early convergence and stabilizes several performance metrics compared with training from scratch.

![Sycamore generalization training](figures/sycamore_generalization_training.png)

Sycamore evaluation savings:

![Sycamore evaluation savings](figures/sycamore_evaluation_savings.png)

Pretrained-policy trajectories under different penetration rates:

![Sycamore pretrained trajectories](figures/sycamore_pretrained_trajectories.png)

## Setup

Create the Python environment:

```bash
conda env create -f environment.yaml
conda activate tdmpc2
```

Install Eclipse SUMO separately and set `SUMO_HOME`.

macOS:

```bash
brew install sumo
export SUMO_HOME=$(brew --prefix sumo)/share/sumo
```

Ubuntu/Debian:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools
export SUMO_HOME=/usr/share/sumo
```

Check the installation:

```bash
sumo --version
python -c "import traci; print('traci OK')"
```

## Quick Checks

Run a basic environment check:

```bash
python test_env.py
```

Run a short training smoke test:

```bash
python train.py steps=3000 eval_freq=3000 save_agent=false ts_plot_freq=0
```

## Training

Train the AV, V2I-only CAV, and V2X CAV settings:

```bash
# AV: no V2I/V2V communication
python train.py exp_name=av_control attention=false communication=false

# CAV-V2I: signal communication only
python train.py exp_name=cav_control_v2i attention=false communication=true

# CAV-V2X: signal and neighboring-CAV communication
python train.py exp_name=cav_control attention=true communication=true
```

The default outputs are written to:

```text
logs/sumo-intersection/<seed>/<exp_name>/
```

The convenience script runs the same core settings:

```bash
bash train.sh
```

## Released Training Artifacts

The curated artifacts follow this layout:

```text
artifacts/training_runs/<experiment>/
├── models/final.pt    Final policy checkpoint
└── train.csv          Episode-level training metrics (400 episodes)
```

The released experiments are:

| Experiment | Role in the paper |
|---|---|
| `av_control` | Local-perception AV policy |
| `cav_control_v2i` | CAV policy with V2I signal information |
| `cav_control` | Proposed Hybrid SAC CAV-V2X policy |
| `cav_control_sac_continuous` | Continuous-action SAC baseline |
| `ppo_control` | Hybrid-action PPO baseline |
| `dqn_control` | 27-action DQN baseline |
| `sycamore_scratch_s1` | Sycamore scenario trained from scratch |
| `sycamore_pretrained_s1` | Sycamore scenario initialized from the pretrained CAV-V2X policy |

The penetration-rate and V2V-context sensitivity evaluations reuse `cav_control/models/final.pt`. The separately trained longitudinal-only and lane-changing-only checkpoints used for the action-branch ablation were unavailable in the local release source and are therefore not included; only their result figure is released. See [`artifacts/README.md`](artifacts/README.md) for file hashes, experiment mapping, and this limitation.

## Evaluation

Evaluate SUMO and GLOSA baselines:

```bash
python evaluate_baseline.py exp_name=sumo_baseline rl_control=false sumo_default_cav_behavior=true
python evaluate_glosa.py exp_name=glosa_baseline glosa_enabled=true
```

Evaluate a trained RL checkpoint:

```bash
python evaluate.py \
  checkpoint=artifacts/training_runs/cav_control/models/final.pt \
  exp_name=cav_control \
  attention=true \
  communication=true
```

Evaluation outputs are written to:

```text
eval_results/<exp_name>/
```

Compare evaluated policies against the SUMO baseline:

```bash
python compare_eval_results_to_baseline.py \
  --eval-root eval_results \
  --baseline sumo_baseline
```

## Real-Parameter Scenario

The repository also includes a Sycamore/Central PM-peak SUMO scenario with real-parameter demand and signal timing represented in the same three-lane through-intersection format.

Train or fine-tune on that scenario:

```bash
bash scripts/train_sycamore_pm2026.sh
```

Run paired scratch/pretrained experiments:

```bash
PRETRAINED_CHECKPOINT=artifacts/training_runs/cav_control/models/final.pt \
bash scripts/train_sycamore_pm2026_scratch_vs_pretrained.sh
```

Visualize a trained policy in SUMO-GUI:

```bash
CHECKPOINT=artifacts/training_runs/cav_control/models/final.pt \
bash scripts/visualize_sycamore_v2x.sh
```

## Configuration Notes

Important Hydra options:

```text
agent: sac | ppo | dqn | tdmpc2
attention: true | false
communication: true | false
comm_topology: full | front_only
context_size: number of neighboring CAV context tokens
communication_range_m: V2V communication range
lane_action_discrete: use categorical lane-change branch for Hybrid SAC
randomize_demand: randomize total flow and CAV penetration during training
```

Use command-line overrides to define fixed evaluation scenarios, for example:

```bash
python evaluate.py checkpoint=<path/to/final.pt> \
  total_flow_vph_min=2000 total_flow_vph_max=2000 \
  cav_penetration_min=0.5 cav_penetration_max=0.5
```

## Large Files

Eight curated `final.pt` checkpoints, their `train.csv` files, and representative paper result figures are tracked. Intermediate checkpoints, runtime `logs/`, raw evaluation results, FCD XML files, videos, and temporary figures remain ignored. The tracked model files are small enough that Git LFS is not required.

## License

MIT License
