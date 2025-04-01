git clone this repository

In the top directory, $ python3 ./src/make_playground.py  
(If you want get help to select other configuration, $ python3 ./src/make_playground.py -h)  
  
A default directory is made like ./playground/0_CartPole-v0_DQN or ./playground/0_CartPole-v0_DQN__X  
  
$ cd ./playground/0_CartPole-v0_DQN  
$ ls -l  
playbatch.sh  playbatch0.sh  postprocess.sh  
$ ./playbatch.sh  
  
playbatch.sh contains playbatch0.sh  
playbatch0.sh contains contains $ python3 play.py  
  
The result is saved in o_XXX_0  
You can check if the running is completed, by $ tail o_XXX_0  
  
After running is completed, the result is post-processed:  
$ ./postprocess.sh  
  
Average returns and its graph are made under a new directory ./playground/0_CartPole-v0_DQN__X/o_XXX and related files are also copied.  
This directory can be moved to ./experiments/  
  
On the other hand, log and model are saved in a directory like ./result/CartPole-v0_DQN_0401_024156  
You can tensorboard like:  
$ tensorboard --logdir ./result/CartPole-v0_DQN_0401_024156/log/summary  
  
You can load the model from the checkpoint ./result/CartPole-v0_DQN_0401_024156/model  

