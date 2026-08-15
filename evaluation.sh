#!/bin/bash
# baseline
python evaluate_baseline.py exp_name=sumo_baseline rl_control=false sumo_default_cav_behavior=true
# SUMO built-in GLOSA baseline equipped on CAVs only
python evaluate_glosa.py exp_name=glosa_baseline glosa_enabled=true
# AV control (no V2X communication, no attention, hide signal remaining-time)
python evaluate.py checkpoint=logs/sumo-intersection/2/av_control/models/final.pt exp_name=av_control attention=false communication=false
# AV + V2I control (no attention, hide signal remaining-time)
python evaluate.py checkpoint=logs/sumo-intersection/2/cav_control_v2i/models/final.pt exp_name=cav_control_v2i attention=false communication=true
# CAV control (full attention, full V2X communication)
python evaluate.py checkpoint=logs/sumo-intersection/2/cav_control/models/final.pt exp_name=cav_control attention=true communication=true
# CAV control (full attention, front V2X communication only)
python evaluate.py checkpoint=logs/sumo-intersection/2/cav_control_front/models/final.pt exp_name=cav_control_front attention=true comm_topology=front_only communication=true
# CAV control (full attention, full V2X communication) + cooperative reward
python evaluate.py checkpoint=logs/sumo-intersection/2/cav_control_coop/models/final.pt exp_name=cav_control_coop attention=true communication=true neighbor_reward_coef=0.10
# CAV control (full attention, full V2X communication) + cooperative reward backcoop
python evaluate.py checkpoint=logs/sumo-intersection/2/cav_control_coop_back/models/final.pt exp_name=cav_control_coop_back attention=true communication=true neighbor_reward_coef=0.10 neighbor_reward_directional=true
