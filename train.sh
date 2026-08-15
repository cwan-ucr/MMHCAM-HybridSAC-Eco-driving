# !/bin/bash
# AV control (no V2X communication, no attention, hide signal remaining-time)
python train.py exp_name=av_control attention=false communication=false
# AV + V2I control (no attention, hide signal remaining-time)
python train.py exp_name=cav_control_v2i attention=false communication=true
# CAV control (full attention, full V2X communication)
python train.py exp_name=cav_control attention=true communication=true
# CAV control (full attention, front V2X communication only)
python train.py exp_name=cav_control_front attention=true comm_topology=front_only communication=true
# CAV control (full attention, full V2X communication) + cooperative reward
python train.py exp_name=cav_control_coop attention=true communication=true neighbor_reward_coef=0.10
# CAV control (full attention, full V2X communication) + cooperative reward only for backward neighbors
python train.py exp_name=cav_control_coop_back attention=true communication=true neighbor_reward_coef=0.10 neighbor_reward_directional=true