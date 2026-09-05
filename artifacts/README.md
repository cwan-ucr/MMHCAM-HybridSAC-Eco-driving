# Released Training Artifacts

This directory contains the final policy checkpoints and episode-level training records used to produce the paper's training comparisons. The files were restored from source commit `24cb5a5` (`All experiment finished`). Each CSV contains one header row and 400 episode rows.

## Experiment Mapping

| Directory | Agent / information setting | Paper usage |
|---|---|---|
| `av_control` | Hybrid SAC, local perception | AV training and evaluation |
| `cav_control_v2i` | Hybrid SAC, V2I | CAV-V2I training and evaluation |
| `cav_control` | Hybrid SAC, V2X with masked cross-attention | Proposed method; action, penetration, and V2V-token studies |
| `cav_control_sac_continuous` | Continuous-action SAC | Algorithm comparison |
| `ppo_control` | Hybrid-action PPO | Algorithm comparison |
| `dqn_control` | DQN with 27 discrete joint actions | Algorithm comparison |
| `sycamore_scratch_s1` | Hybrid SAC trained from scratch | Data-informed scenario transfer comparison |
| `sycamore_pretrained_s1` | Hybrid SAC initialized from `cav_control` | Data-informed scenario transfer comparison |

The penetration-rate and V2V-context sensitivity experiments evaluate the unchanged `cav_control/models/final.pt` policy while changing the evaluation scenario or masking context tokens. The separately trained longitudinal-only and lane-changing-only checkpoints used for the action-branch ablation were not available in the local release source. Those two policies cannot currently be re-evaluated bit-for-bit from the released checkpoints.

## File Layout

```text
training_runs/<experiment>/
├── models/final.pt
└── train.csv
```

Checkpoints are PyTorch dictionaries with a top-level `model` state dictionary. Load a checkpoint through the repository's agent/evaluation code rather than unpickling files from untrusted sources.

## SHA-256 Checksums

| File | SHA-256 |
|---|---|
| `training_runs/av_control/models/final.pt` | `3eb33ed962f39ec6a0a3bc55cb66dd017f982a45be080eb1387374d075ccb290` |
| `training_runs/av_control/train.csv` | `7b7debb378c983281f60ef189ceeb949db857818156b0ec17e78b7c086384bf1` |
| `training_runs/cav_control/models/final.pt` | `e7cd0a28163debf35ca112d3985065c73172f0836664dc4ec75bad53b4ef4018` |
| `training_runs/cav_control/train.csv` | `8fe5807d7af70552db5c3182c464e5b349e9971d7ffc36b18d86d0948a42576e` |
| `training_runs/cav_control_sac_continuous/models/final.pt` | `953974f248ce1a26d7be2d145918d1842134ff1cc610e5e76b233ccee3f84537` |
| `training_runs/cav_control_sac_continuous/train.csv` | `f87b09e5fcdbcbc5685acd81b4a00894cb6fcd22db9001ebc5fe12925fbf979f` |
| `training_runs/cav_control_v2i/models/final.pt` | `194c02d41f0a35a112693f14a57611a6e419c12876d0416ecc49ded173043bf1` |
| `training_runs/cav_control_v2i/train.csv` | `cd3e5ad0dbcca6a1fb3691a4b5a8a1437a634c94e4bed9571cf0e0eddfce0f5` |
| `training_runs/dqn_control/models/final.pt` | `2fb572b1779d9167d3d392aa4b2aab1bfa0a18cf1d3cd1f58402a307177cb9c6` |
| `training_runs/dqn_control/train.csv` | `ac05edbce07bb3e43a4dde016b486f2a72504e6e7f34e81455bad3f3c24caa3c` |
| `training_runs/ppo_control/models/final.pt` | `699dad4d8a757dc3f434f83c90c9e5c5f6108aab82f55e8ac636cad4ee67c253` |
| `training_runs/ppo_control/train.csv` | `a18483c1a2764e1db192cdc4067b5c63ee738116d8a2f03ac2a188e3fa47037e` |
| `training_runs/sycamore_pretrained_s1/models/final.pt` | `e265cdddf932ad5b0a7ac2e0c8471ce1c3c3fdd226fb9b96621d38e087e16547` |
| `training_runs/sycamore_pretrained_s1/train.csv` | `050d6b023ffcd6346c7f3c2454367919da614ef9f5744357c071091e8554807d` |
| `training_runs/sycamore_scratch_s1/models/final.pt` | `61d5bcc026e930058b78a6d40a7486117649072273aa87c587cf069f0ce8d304` |
| `training_runs/sycamore_scratch_s1/train.csv` | `0e423f1e8a5b4da8107017bb506b856459073c8c85ac17297f20e35f1b63111d` |
