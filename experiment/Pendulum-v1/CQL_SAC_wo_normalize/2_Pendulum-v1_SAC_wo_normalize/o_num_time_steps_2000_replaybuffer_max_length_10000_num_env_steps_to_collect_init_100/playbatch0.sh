#!/bin/bash

cd /home/soh/work/try_tf_agents/src
python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -t 2000 -l 10000 -i 100 -c /home/soh/work/try_tf_agents/result/Pendulum-v1_CQL_SAC_wo_normalize_0221_054204/model &> /home/soh/work/try_tf_agents/playground/Pendulum-v1_SAC_wo_normalize_2/o_num_time_steps_2000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_100_0 &
