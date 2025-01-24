#!/bin/bash

# python3 play.py -e CartPole-v0 -a TD3 -i 100 &> o_CartPole-v0_TD3_num_init_collect_steps_100_0 ; \
# python3 play.py -e CartPole-v0 -a TD3 -i 200 &> o_CartPole-v0_TD3_num_init_collect_steps_200_0 ; \
# python3 play.py -e CartPole-v0 -a TD3 -i 500 &> o_CartPole-v0_TD3_num_init_collect_steps_500_0 ; \
# python3 play.py -e CartPole-v0 -a TD3 -i 1000 &> o_CartPole-v0_TD3_num_init_collect_steps_1000_0 ; \
# python3 play.py -e CartPole-v0 -a TD3 -i 2000 &> o_CartPole-v0_TD3_num_init_collect_steps_2000_0 ; \
# python3 play.py -e CartPole-v0 -a TD3 -i 5000 &> o_CartPole-v0_TD3_num_init_collect_steps_5000_0 ; \
# python3 play.py -e CartPole-v0 -a TD3 -i 10000 &> o_CartPole-v0_TD3_num_init_collect_steps_10000_0 &

# python3 play.py -e CartPole-v0 -a CDQN -n 5 &> o_CartPole-v0_CDQN_num_actions_5_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -n 7 &> o_CartPole-v0_CDQN_num_actions_7_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN_multiagent   &> o_CartPole-v0_CDQN_multiagent_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN_multiagent -n 5 &> o_CartPole-v0_CDQN_multiagent_num_actions_5_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN_multiagent -n 7 &> o_CartPole-v0_CDQN_multiagent_num_actions_7_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN &> o_CartPole-v0_CDQN_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   &> o_CartPole-v0_CDQN_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -n 5 &> o_CartPole-v0_CDQN_num_actions_5_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -n 7 &> o_CartPole-v0_CDQN_num_actions_7_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN_multiagent   &> o_CartPole-v0_CDQN_multiagent_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN_multiagent -n 5 &> o_CartPole-v0_CDQN_multiagent_num_actions_5_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN_multiagent -n 7 &> o_CartPole-v0_CDQN_multiagent_num_actions_7_0 &

# python3 play.py -e CartPole-v0 -a CDQN -i 100 &> o_CartPole-v0_CDQN_num_init_collect_steps_100_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -i 200 &> o_CartPole-v0_CDQN_num_init_collect_steps_200_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -i 500 &> o_CartPole-v0_CDQN_num_init_collect_steps_500_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -i 1000 &> o_CartPole-v0_CDQN_num_init_collect_steps_1000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -i 2000 &> o_CartPole-v0_CDQN_num_init_collect_steps_2000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -i 5000 &> o_CartPole-v0_CDQN_num_init_collect_steps_5000_0 &

# python3 play.py -e CartPole-v0 -a CDQN   -i 100 -g 0.01 &> o_CartPole-v0_CDQN_num_init_collect_steps_100_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 200 -g 0.01 &> o_CartPole-v0_CDQN_num_init_collect_steps_200_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 500 -g 0.01 &> o_CartPole-v0_CDQN_num_init_collect_steps_500_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 1000 -g 0.01 &> o_CartPole-v0_CDQN_num_init_collect_steps_1000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 2000 -g 0.01 &> o_CartPole-v0_CDQN_num_init_collect_steps_2000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 5000 -g 0.01 &> o_CartPole-v0_CDQN_num_init_collect_steps_5000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 10000 -g 0.01 &> o_CartPole-v0_CDQN_num_init_collect_steps_10000_0 &

# python3 play.py -e CartPole-v0 -a CDQN   -i 100 &> o_CartPole-v0_CDQN_num_init_collect_steps_100_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 200 &> o_CartPole-v0_CDQN_num_init_collect_steps_200_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 500 &> o_CartPole-v0_CDQN_num_init_collect_steps_500_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 1000 &> o_CartPole-v0_CDQN_num_init_collect_steps_1000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN   -i 2000 &> o_CartPole-v0_CDQN_num_init_collect_steps_2000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -c ./result/CartPole-v0_CDQN_1216_081611/model -f true -i 2000 &> o_CartPole-v0_CDQN_num_init_collect_steps_2000_0 ; \

python3 play.py -e CartPole-v0 -a CDQN -i 5000 &> o_CartPole-v0_CDQN_num_init_collect_steps_5000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -i 10000 &> o_CartPole-v0_CDQN_num_init_collect_steps_10000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -i 20000 &> o_CartPole-v0_CDQN_num_init_collect_steps_20000_0 ; \
# python3 play.py -e CartPole-v0 -a CDQN -i 50000 &> o_CartPole-v0_CDQN_num_init_collect_steps_50000_0 &

