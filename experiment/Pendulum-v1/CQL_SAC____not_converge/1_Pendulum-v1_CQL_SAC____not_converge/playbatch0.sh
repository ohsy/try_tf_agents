#!/bin/bash

cd /home/soh/work/try_tf_agents/src
python3 play.py -e Pendulum-v1 -a CQL_SAC -t 20000 -l 10000 -c /home/soh/work/try_tf_agents/result/Pendulum-v1_SAC_0221_030448/model &> /home/soh/work/try_tf_agents/playground/Pendulum-v1_CQL_SAC_11/o_num_time_steps_20000_replaybuffer_max_length_10000_0 &
