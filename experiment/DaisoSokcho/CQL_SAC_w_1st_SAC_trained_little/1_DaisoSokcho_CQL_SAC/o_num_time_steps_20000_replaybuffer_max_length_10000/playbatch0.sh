#!/bin/bash

cd /home/soh/work/try_tf_agents/src
python3 play.py -e DaisoSokcho -a CQL_SAC -t 20000 -l 10000 -c /home/soh/work/try_tf_agents/result/DaisoSokcho_SAC_0222_120638/model &> /home/soh/work/try_tf_agents/playground/DaisoSokcho_CQL_SAC_11/o_num_time_steps_20000_replaybuffer_max_length_10000_0 &
