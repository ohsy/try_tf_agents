#!/bin/bash

# python3 play.py -e Reacher-v2_discrete -a DQN_multiagent -n 5 &> o_Reacher-v2_discrete_DQN_multiagent_num_actions_5_3 ; \
# python3 play.py -e Reacher-v2_discrete -a DQN_multiagent -n 7 &> o_Reacher-v2_discrete_DQN_multiagent_num_actions_7_3 ; \
# python3 play.py -e DaisoSokcho -a SAC &> o_DaisoSokcho_SAC_3 ; \
python3 play.py -e DaisoSokcho_discrete -a CDQN_multiagent -n 3 &> o_DaisoSokcho_discrete_CDQN_multiagent_num_actions_3_3 ; \
python3 play.py -e DaisoSokcho_discrete -a CDQN_multiagent -n 5 &> o_DaisoSokcho_discrete_CDQN_multiagent_num_actions_5_3 ; \
python3 play.py -e DaisoSokcho_discrete -a CDQN_multiagent -n 7 &> o_DaisoSokcho_discrete_CDQN_multiagent_num_actions_7_3 ; \
python3 play.py -e DaisoSokcho_discrete -a DQN_multiagent -n 3 &> o_DaisoSokcho_discrete_DQN_multiagent_num_actions_3_3 ; \
python3 play.py -e DaisoSokcho_discrete -a DQN_multiagent -n 5 &> o_DaisoSokcho_discrete_DQN_multiagent_num_actions_5_3 ; \
python3 play.py -e DaisoSokcho_discrete -a DQN_multiagent -n 7 &> o_DaisoSokcho_discrete_DQN_multiagent_num_actions_7_3 &
