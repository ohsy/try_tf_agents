#!/bin/bash

# python3 play.py -e DaisoSokcho -a TD3 -i 100 &> o_DaisoSokcho_TD3_num_init_collect_steps_100_4 ; \
# python3 play.py -e DaisoSokcho -a TD3 -i 200 &> o_DaisoSokcho_TD3_num_init_collect_steps_200_4 ; \
# python3 play.py -e DaisoSokcho -a TD3 -i 500 &> o_DaisoSokcho_TD3_num_init_collect_steps_500_4 ; \
# python3 play.py -e DaisoSokcho -a TD3 -i 1000 &> o_DaisoSokcho_TD3_num_init_collect_steps_1000_4 ; \
# python3 play.py -e DaisoSokcho -a TD3 -i 2000 &> o_DaisoSokcho_TD3_num_init_collect_steps_2000_4 ; \
# python3 play.py -e DaisoSokcho -a TD3 -i 5000 &> o_DaisoSokcho_TD3_num_init_collect_steps_5000_4 ; \
# python3 play.py -e DaisoSokcho -a TD3 -i 10000 &> o_DaisoSokcho_TD3_num_init_collect_steps_10000_4 &

# python3 play.py -e DaisoSokcho -a CQL_SAC -n 5 &> o_DaisoSokcho_CQL_SAC_num_actions_5_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -n 7 &> o_DaisoSokcho_CQL_SAC_num_actions_7_4 ; \
# python3 play.py -e DaisoSokcho -a DQN_multiagent   &> o_DaisoSokcho_DQN_multiagent_4 ; \
# python3 play.py -e DaisoSokcho -a DQN_multiagent -n 5 &> o_DaisoSokcho_DQN_multiagent_num_actions_5_4 ; \
# python3 play.py -e DaisoSokcho -a DQN_multiagent -n 7 &> o_DaisoSokcho_DQN_multiagent_num_actions_7_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC &> o_DaisoSokcho_CQL_SAC_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   &> o_DaisoSokcho_CQL_SAC_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -n 5 &> o_DaisoSokcho_CQL_SAC_num_actions_5_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -n 7 &> o_DaisoSokcho_CQL_SAC_num_actions_7_4 ; \
# python3 play.py -e DaisoSokcho -a DQN_multiagent   &> o_DaisoSokcho_DQN_multiagent_4 ; \
# python3 play.py -e DaisoSokcho -a DQN_multiagent -n 5 &> o_DaisoSokcho_DQN_multiagent_num_actions_5_4 ; \
# python3 play.py -e DaisoSokcho -a DQN_multiagent -n 7 &> o_DaisoSokcho_DQN_multiagent_num_actions_7_4 &

# python3 play.py -e DaisoSokcho -a CQL_SAC -i 100 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_100_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -i 200 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_200_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -i 500 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_500_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -i 1000 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_1000_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -i 2000 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_2000_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -i 5000 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_5000_4 &

# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 100 -g 0.01 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_100_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 200 -g 0.01 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_200_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 500 -g 0.01 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_500_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 1000 -g 0.01 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_1000_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 2000 -g 0.01 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_2000_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 5000 -g 0.01 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_5000_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 10000 -g 0.01 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_10000_4 &

# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 100 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_100_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 200 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_200_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 500 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_500_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 1000 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_1000_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC   -i 2000 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_2000_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -c ./result/DaisoSokcho_CQL_SAC_1216_081611/model -f true -i 2000 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_2000_4 ; \

python3 play.py -e DaisoSokcho -a CQL_SAC -c ./result/DaisoSokcho_SAC_0125_072713/model &> o_DaisoSokcho_CQL_SAC_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -i 10000 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_10000_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -i 20000 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_20000_4 ; \
# python3 play.py -e DaisoSokcho -a CQL_SAC -i 50000 &> o_DaisoSokcho_CQL_SAC_num_init_collect_steps_50000_4 &

