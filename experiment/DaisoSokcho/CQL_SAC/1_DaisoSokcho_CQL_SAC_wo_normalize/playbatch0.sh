#!/bin/bash

cd /home/soh/work/try_tf_agents/src
python3 play.py -e DaisoSokcho -a CQL_SAC_wo_normalize -t 5000 -l 10000 -c /home/soh/work/try_tf_agents/result/DaisoSokcho_SAC_0221_051416/model &> /home/soh/work/try_tf_agents/playground/DaisoSokcho_CQL_SAC_wo_normalize_1/o_num_time_steps_5000_replaybuffer_max_length_10000_0 &
