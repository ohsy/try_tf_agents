#!/bin/bash

# python3 play.py -e Reacher-v2 -a SAC &> o_Reacher-v2_SAC_4 ; \
# python3 play.py -e Reacher-v2_discrete -a CDQN_multiagent -n 3 &> o_Reacher-v2_discrete_CDQN_multiagent_num_actions_3_4 ; \
# python3 play.py -e Reacher-v2_discrete -a CDQN_multiagent -n 5 &> o_Reacher-v2_discrete_CDQN_multiagent_num_actions_5_4 ; \
# python3 play.py -e Reacher-v2_discrete -a CDQN_multiagent -n 7 &> o_Reacher-v2_discrete_CDQN_multiagent_num_actions_7_4 ; \
# python3 play.py -e Reacher-v2_discrete -a DQN_multiagent -n 3 &> o_Reacher-v2_discrete_DQN_multiagent_num_actions_3_4 ; \
# python3 play.py -e Reacher-v2_discrete -a DQN_multiagent -n 5 &> o_Reacher-v2_discrete_DQN_multiagent_num_actions_5_4 ; \
# python3 play.py -e Reacher-v2_discrete -a DQN_multiagent -n 7 &> o_Reacher-v2_discrete_DQN_multiagent_num_actions_7_4 ; \
# python3 play.py -e DaisoSokcho -a SAC &> o_DaisoSokcho_SAC_4 ; \
# python3 play.py -e DaisoSokcho_discrete -a CDQN_multiagent -n 3 &> o_DaisoSokcho_discrete_CDQN_multiagent_num_actions_3_4 ; \
# python3 play.py -e DaisoSokcho_discrete -a CDQN_multiagent -n 5 &> o_DaisoSokcho_discrete_CDQN_multiagent_num_actions_5_4 ; \
# python3 play.py -e DaisoSokcho_discrete -a CDQN_multiagent -n 7 &> o_DaisoSokcho_discrete_CDQN_multiagent_num_actions_7_4 ; \
# python3 play.py -e DaisoSokcho_discrete -a DQN_multiagent -n 3 &> o_DaisoSokcho_discrete_DQN_multiagent_num_actions_3_4 ; \
python3 play.py -e DaisoSokcho_discrete -a DQN_multiagent -n 5 &> o_DaisoSokcho_discrete_DQN_multiagent_num_actions_5_4 ; \
python3 play.py -e DaisoSokcho_discrete -a DQN_multiagent -n 7 &> o_DaisoSokcho_discrete_DQN_multiagent_num_actions_7_4 &
