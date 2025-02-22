#!/bin/bash

# python3 get_avg_returns.py 5 o_CartPole-v0_DQN_num_init_collect_steps_100
# python3 plot_csv.py o_CartPole-v0_DQN_num_init_collect_steps_100
# mkdir o_CartPole-v0_DQN_num_init_collect_steps_100
# mv o_CartPole-v0_DQN_num_init_collect_steps_100.* o_CartPole-v0_DQN_num_init_collect_steps_100
# mv o_CartPole-v0_DQN_num_init_collect_steps_100_* o_CartPole-v0_DQN_num_init_collect_steps_100

# python3 get_avg_returns.py 5 o_CartPole-v0_DQN_num_init_collect_steps_200
# python3 plot_csv.py o_CartPole-v0_DQN_num_init_collect_steps_200
# mkdir o_CartPole-v0_DQN_num_init_collect_steps_200
# mv o_CartPole-v0_DQN_num_init_collect_steps_200.* o_CartPole-v0_DQN_num_init_collect_steps_200
# mv o_CartPole-v0_DQN_num_init_collect_steps_200_* o_CartPole-v0_DQN_num_init_collect_steps_200

# python3 get_avg_returns.py 5 o_CartPole-v0_DQN_num_init_collect_steps_500
# python3 plot_csv.py o_CartPole-v0_DQN_num_init_collect_steps_500
# mkdir o_CartPole-v0_DQN_num_init_collect_steps_500
# mv o_CartPole-v0_DQN_num_init_collect_steps_500.* o_CartPole-v0_DQN_num_init_collect_steps_500
# mv o_CartPole-v0_DQN_num_init_collect_steps_500_* o_CartPole-v0_DQN_num_init_collect_steps_500

# python3 get_avg_returns.py 5 o_CartPole-v0_DQN_num_init_collect_steps_1000
# python3 plot_csv.py o_CartPole-v0_DQN_num_init_collect_steps_1000
# mkdir o_CartPole-v0_DQN_num_init_collect_steps_1000
# mv o_CartPole-v0_DQN_num_init_collect_steps_1000.* o_CartPole-v0_DQN_num_init_collect_steps_1000
# mv o_CartPole-v0_DQN_num_init_collect_steps_1000_* o_CartPole-v0_DQN_num_init_collect_steps_1000

# python3 get_avg_returns.py 5 o_CartPole-v0_DQN_num_init_collect_steps_2000
# python3 plot_csv.py o_CartPole-v0_DQN_num_init_collect_steps_2000
# mkdir o_CartPole-v0_DQN_num_init_collect_steps_2000
# mv o_CartPole-v0_DQN_num_init_collect_steps_2000.* o_CartPole-v0_DQN_num_init_collect_steps_2000
# mv o_CartPole-v0_DQN_num_init_collect_steps_2000_* o_CartPole-v0_DQN_num_init_collect_steps_2000

python3 get_avg_returns.py 5 o_CartPole-v0_DQN
python3 plot_csv.py o_CartPole-v0_DQN
mkdir o_CartPole-v0_DQN
mv o_CartPole-v0_DQN.* o_CartPole-v0_DQN
mv o_CartPole-v0_DQN_* o_CartPole-v0_DQN
cp config.json playbatch*.sh postprocess.sh o_CartPole-v0_DQN

# python3 get_avg_returns.py 5 o_CartPole-v0_DQN_num_init_collect_steps_10000
# python3 plot_csv.py o_CartPole-v0_DQN_num_init_collect_steps_10000
# mkdir o_CartPole-v0_DQN_num_init_collect_steps_10000
# mv o_CartPole-v0_DQN_num_init_collect_steps_10000.* o_CartPole-v0_DQN_num_init_collect_steps_10000
# mv o_CartPole-v0_DQN_num_init_collect_steps_10000_* o_CartPole-v0_DQN_num_init_collect_steps_10000

