#!/bin/bash

env=$1
agent=$2
num_actions=$3
k=$4
file_prefix=o_${env}_${agent}_num_actions_${num_actions}
# echo $env
# echo $agent
# echo $num_actions
# echo $k
# echo $file_prefix

for (( i=0; i < $k; ++i ))
do 
 	python3 play.py -e $env -a $agent -n $num_actions &> ${file_prefix}_${i} &
done

# python3 get_avg_returns.py $k $file_prefix
