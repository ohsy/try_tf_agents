
from __future__ import absolute_import, division, print_function

import functools
import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import base64
import imageio
# import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import PIL.Image
# import pyvirtualdisplay
import reverb
import time

from absl import app

import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_gym

from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network

from tf_agents.drivers import py_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.system import system_multiprocessing as multiprocessing


def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


def main(_):
    # env_name = "MinitaurBulletEnv-v0" # @param {type:"string"}
    # env_name = "HalfCheetah-v2" # @param {type:"string"}
    env_name = "Pendulum-v1" # @param {type:"string"}
    # Use "num_iterations = 1e6" for better results (2 hrs)
    # 1e5 is just so this doesn't take too long (1 hr)
    # num_iterations = 100000 # 20000 # @param {type:"integer"}
    num_iterations = 200000 # 20000 # @param {type:"integer"}

    initial_collect_steps = 20000  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    # replay_buffer_max_length = 10000  # @param {type:"integer"}
    replay_buffer_max_length = 1001  # @param {type:"integer"}

    batch_size = 256  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}

    actor_fc_layer_params = (256, 128)
    value_fc_layer_params = (256, 128)

    log_interval = 200 # @param {type:"integer"}

    num_eval_episodes = 10 # @param {type:"integer"}
    eval_interval = 1000 # @param {type:"integer"}

    policy_save_interval = 5000 # @param {type:"integer"}

    num_parallel_environments = 30
    num_environment_steps = 25000000
    num_epochs = 25
    collect_episodes_per_iteration = num_parallel_environments
    num_eval_episodes = 10

    debug_summaries=False
    summarize_grads_and_vars=False


    tf_train_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [lambda: suite_gym.load(env_name)] * num_parallel_environments
        )
    )
    tf_eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

#     observation_spec, action_spec, time_step_spec = (
#         spec_utils.get_tensor_specs(train_env))
    tf_observation_spec = tf_train_env.observation_spec()
    tf_action_spec = tf_train_env.action_spec()
    tf_time_step_spec = tf_train_env.time_step_spec()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_observation_spec,
        tf_action_spec,
        fc_layer_params=actor_fc_layer_params,
        activation_fn=tf.keras.activations.tanh,)
    value_net = value_network.ValueNetwork(
        tf_observation_spec,
        fc_layer_params=value_fc_layer_params,
        activation_fn=tf.keras.activations.tanh,)
    agent = ppo_clip_agent.PPOClipAgent(
        tf_time_step_spec,
        tf_action_spec,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=num_epochs,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_utils.create_train_step())
    agent.initialize()

    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [tf_metrics.NumberOfEpisodes(), environment_steps_metric,]
    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(batch_size=num_parallel_environments),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=num_parallel_environments),]

    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(tf_time_step_spec, tf_action_spec)

    random_return = compute_avg_return(tf_eval_env, random_policy, num_eval_episodes)
    print(f"random policy return = {random_return}", flush=True)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_buffer_max_length,
    )

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(tf.Variable(0, dtype=tf.int64))

    avg_return = compute_avg_return(tf_eval_env, agent.policy, num_eval_episodes)
    print(f"before training, agent policy return = {avg_return}", flush=True)
    returns = [avg_return]

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_train_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration,
    )

    tf_train_env.reset()

    before = time.time()
    for _ in range(num_iterations):

        collect_driver.run()

        trajectories = replay_buffer.gather_all()
        train_loss = agent.train(experience=trajectories).loss
        replay_buffer.clear()

        train_step = agent.train_step_counter.numpy()

        if train_step % log_interval == 0:
            after = time.time()
            print(f'train_step = {train_step}: loss = {train_loss}, time used = {after - before}', flush=True)
            before = after

        if train_step % eval_interval == 0:
            avg_return = compute_avg_return(tf_eval_env, agent.policy, num_eval_episodes)
            print(f'train_step = {train_step}: average return = {avg_return}', flush=True)
            returns.append(avg_return)


if __name__ == '__main__':
  # flags.mark_flag_as_required('root_dir')
  multiprocessing.handle_main(functools.partial(app.run, main))

