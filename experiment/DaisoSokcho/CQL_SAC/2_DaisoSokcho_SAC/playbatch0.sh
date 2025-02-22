#!/bin/bash

cd /home/soh/work/try_tf_agents/src
python3 play.py -e DaisoSokcho -a SAC -t 2000 -l 10000 -i 100 -c /home/soh/work/try_tf_agents/result/DaisoSokcho_CQL_SAC_0221_101005/model &> /home/soh/work/try_tf_agents/playground/DaisoSokcho_SAC_2/o_num_time_steps_2000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_100_0 &
