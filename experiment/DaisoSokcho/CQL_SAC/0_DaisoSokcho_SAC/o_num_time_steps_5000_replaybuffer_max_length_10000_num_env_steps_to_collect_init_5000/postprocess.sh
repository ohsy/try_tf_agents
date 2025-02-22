#!/bin/bash

cd /home/soh/work/try_tf_agents/src
python3 get_avg_returns.py 1 /home/soh/work/try_tf_agents/playground/DaisoSokcho_SAC_0/o_num_time_steps_5000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_5000
python3 plot_csv.py /home/soh/work/try_tf_agents/playground/DaisoSokcho_SAC_0/o_num_time_steps_5000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_5000
cd /home/soh/work/try_tf_agents/playground/DaisoSokcho_SAC_0
mkdir o_num_time_steps_5000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_5000
mv o_num_time_steps_5000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_5000.* o_num_time_steps_5000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_5000
mv o_num_time_steps_5000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_5000_* o_num_time_steps_5000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_5000
cp /home/soh/work/try_tf_agents/src/play.py /home/soh/work/try_tf_agents/src/game.py /home/soh/work/try_tf_agents/src/config.json playbatch*.sh postprocess.sh o_num_time_steps_5000_replaybuffer_max_length_10000_num_env_steps_to_collect_init_5000
