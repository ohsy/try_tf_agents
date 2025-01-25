#!/bin/bash

# python3 play.py -e Pendulum-v1 -a TD3 -i 100 &> o_Pendulum-v1_TD3_num_init_collect_steps_100_4 ; \
# python3 play.py -e Pendulum-v1 -a TD3 -i 200 &> o_Pendulum-v1_TD3_num_init_collect_steps_200_4 ; \
# python3 play.py -e Pendulum-v1 -a TD3 -i 500 &> o_Pendulum-v1_TD3_num_init_collect_steps_500_4 ; \
# python3 play.py -e Pendulum-v1 -a TD3 -i 1000 &> o_Pendulum-v1_TD3_num_init_collect_steps_1000_4 ; \
# python3 play.py -e Pendulum-v1 -a TD3 -i 2000 &> o_Pendulum-v1_TD3_num_init_collect_steps_2000_4 ; \
# python3 play.py -e Pendulum-v1 -a TD3 -i 5000 &> o_Pendulum-v1_TD3_num_init_collect_steps_5000_4 ; \
# python3 play.py -e Pendulum-v1 -a TD3 -i 10000 &> o_Pendulum-v1_TD3_num_init_collect_steps_10000_4 &

# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -n 5 &> o_Pendulum-v1_SAC_wo_normalize_num_actions_5_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -n 7 &> o_Pendulum-v1_SAC_wo_normalize_num_actions_7_4 ; \
# python3 play.py -e Pendulum-v1 -a DQN_multiagent   &> o_Pendulum-v1_DQN_multiagent_4 ; \
# python3 play.py -e Pendulum-v1 -a DQN_multiagent -n 5 &> o_Pendulum-v1_DQN_multiagent_num_actions_5_4 ; \
# python3 play.py -e Pendulum-v1 -a DQN_multiagent -n 7 &> o_Pendulum-v1_DQN_multiagent_num_actions_7_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize &> o_Pendulum-v1_SAC_wo_normalize_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   &> o_Pendulum-v1_SAC_wo_normalize_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -n 5 &> o_Pendulum-v1_SAC_wo_normalize_num_actions_5_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -n 7 &> o_Pendulum-v1_SAC_wo_normalize_num_actions_7_4 ; \
# python3 play.py -e Pendulum-v1 -a DQN_multiagent   &> o_Pendulum-v1_DQN_multiagent_4 ; \
# python3 play.py -e Pendulum-v1 -a DQN_multiagent -n 5 &> o_Pendulum-v1_DQN_multiagent_num_actions_5_4 ; \
# python3 play.py -e Pendulum-v1 -a DQN_multiagent -n 7 &> o_Pendulum-v1_DQN_multiagent_num_actions_7_4 &

# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -i 100 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_100_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -i 200 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_200_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -i 500 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_500_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -i 1000 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_1000_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -i 2000 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_2000_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -i 5000 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_5000_4 &

# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 100 -g 0.01 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_100_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 200 -g 0.01 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_200_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 500 -g 0.01 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_500_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 1000 -g 0.01 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_1000_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 2000 -g 0.01 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_2000_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 5000 -g 0.01 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_5000_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 10000 -g 0.01 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_10000_4 &

# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 100 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_100_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 200 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_200_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 500 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_500_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 1000 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_1000_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize   -i 2000 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_2000_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -c ./result/Pendulum-v1_SAC_wo_normalize_1216_081611/model -f true -i 2000 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_2000_4 ; \

python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -c ./result/Pendulum-v1_CQL_SAC_0125_123106/model &> o_Pendulum-v1_SAC_wo_normalize_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -i 10000 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_10000_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -i 20000 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_20000_4 ; \
# python3 play.py -e Pendulum-v1 -a SAC_wo_normalize -i 50000 &> o_Pendulum-v1_SAC_wo_normalize_num_init_collect_steps_50000_4 &

