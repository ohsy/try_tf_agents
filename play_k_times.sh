#!/bin/bash

env=$1
agent=$2
k=$3
file_prefix=o_${env}_${agent}

for (( i=0; i < $k; ++i ))
do 
 	python3 play.py -e $env -a $agent &> ${file_prefix}_${i} &
done

# python3 get_avg_returns.py $k $file_prefix
