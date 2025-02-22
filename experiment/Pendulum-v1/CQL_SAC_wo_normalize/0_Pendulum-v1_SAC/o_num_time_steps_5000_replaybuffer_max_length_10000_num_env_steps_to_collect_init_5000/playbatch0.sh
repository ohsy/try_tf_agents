#!/bin/bash

cd /home/soh/work/try_tf_agents/src
python3 play.py -e Pendulum-v1 -a SAC -t 5000 -l 10000 -i 5000 &> /home/soh/work/try_tf_agents/playground/Pendulum-v1_SAC_0/o_num_time_steps_5000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_5000_0 &
