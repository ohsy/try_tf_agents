
from __future__ import absolute_import, division, print_function

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

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
# from tf_agents.environments import suite_pybullet
# from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network

# from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_step_driver
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
from tf_agents.policies import policy_saver


# tempdir = tempfile.gettempdir()


# Set up a virtual display for rendering OpenAI gym environments.
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

# env_name = "MinitaurBulletEnv-v0" # @param {type:"string"}
env_name = "Pendulum-v1" # @param {type:"string"}
# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
# num_iterations = 100000 # 20000 # @param {type:"integer"}
num_iterations = 5000 # 20000 # @param {type:"integer"}

initial_collect_steps = 10000  # @param {type:"integer"}
collect_steps_per_iteration = 1# @param {type:"integer"}
replay_buffer_max_length = 10000  # @param {type:"integer"}

batch_size = 256  # @param {type:"integer"}
critic_learning_rate = 6e-4  # @param {type:"number"}
actor_learning_rate = 3e-4  # @param {type:"number"}
alpha_learning_rate = 3e-4  # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 200 # @param {type:"integer"}

num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 1000 # @param {type:"integer"}

# policy_save_interval = 5000 # @param {type:"integer"}

# env = suite_gym.load(env_name)
# env.reset()
# PIL.Image.fromarray(env.render())

# collect_env = suite_pybullet.load(env_name)
# eval_env = suite_pybullet.load(env_name)
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(train_env))
print('Observation Spec:')
print(observation_spec)
print('Action Spec:')
print(action_spec)
print('Reward Spec:')
print(time_step_spec.reward)

critic_net = critic_network.CriticNetwork(
      (observation_spec, action_spec),
      observation_fc_layer_params=None,
      action_fc_layer_params=None,
      joint_fc_layer_params=critic_joint_fc_layer_params,
      kernel_initializer='glorot_uniform',
      last_kernel_initializer='glorot_uniform')

actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=(
          tanh_normal_projection_network.TanhNormalProjectionNetwork))

train_step = train_utils.create_train_step()

agent = sac_agent.SacAgent(
      time_step_spec,
      action_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.keras.optimizers.Adam(
          learning_rate=actor_learning_rate),
      critic_optimizer=tf.keras.optimizers.Adam(
          learning_rate=critic_learning_rate),
      alpha_optimizer=tf.keras.optimizers.Adam(
          learning_rate=alpha_learning_rate),
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      train_step_counter=train_step)

agent.initialize()

# eval_policy = agent.policy
# collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())
# random_policy = random_py_policy.RandomPyPolicy(
#     collect_env.time_step_spec(), collect_env.action_spec())

# example_environment = tf_py_environment.TFPyEnvironment(
#     suite_gym.load(env_name))
# time_step = example_environment.reset()
# random_policy.action(time_step)

#@test {"skip": true}
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


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

compute_avg_return(eval_env, random_policy, num_eval_episodes)

table_name = 'uniform_table'
"""
replay_buffer_signature = tensor_spec.from_spec(
    agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2,
  stride_length=1)
"""

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)
print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
iterator = iter(dataset)
print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)
print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)


checkpoint_path = os.path.join(
    os.path.abspath(os.getcwd()),
    f'result/checkpoint'
)
print(f"before checkpointer is instantiated")
print(f"dataset.cardinality()={dataset.cardinality()}", flush=True)
print(f"replay_buffer.capacity={replay_buffer.capacity}", flush=True)
print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)

train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_path,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=agent.train_step_counter
)
print(f"before checkpointer.initialize_or_restore")
print(f"dataset.cardinality()={dataset.cardinality()}", flush=True)
print(f"replay_buffer.capacity={replay_buffer.capacity}", flush=True)
print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)

train_checkpointer.initialize_or_restore()  # not necessary??
print(f"after checkpointer.initialize_or_restore")
print(f"dataset.cardinality()={dataset.cardinality()}", flush=True)
print(f"replay_buffer.capacity={replay_buffer.capacity}", flush=True)
print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
# time_step = train_env.reset()
time_step = train_py_env.reset()

before = time.time()
for _ in range(num_iterations):

  # Collect a few steps and save to the replay buffer.
  # time_step, _ = collect_driver.run(time_step=time_step)
  collect_driver.run()

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))
    after = time.time()
    print('time used = {0}'.format(after - before))
    before = after

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

# print(dir(agent.policy))

#@test {"skip": true}

# iterations = range(0, num_iterations + 1, eval_interval)
# plt.plot(iterations, returns)
# plt.ylabel('Average Return')
# plt.xlabel('Iterations')
# plt.ylim(top=250)


# def embed_mp4(filename):
#   """Embeds an mp4 file in the notebook."""
#   video = open(filename,'rb').read()
#   b64 = base64.b64encode(video)
#   tag = '''
#   '''.format(b64.decode())
# 
#   return IPython.display.HTML(tag)

# def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
#   filename = filename + ".mp4"
#   with imageio.get_writer(filename, fps=fps) as video:
#     for _ in range(num_episodes):
#       time_step = eval_env.reset()
#       video.append_data(eval_py_env.render())
#       while not time_step.is_last():
#         action_step = policy.action(time_step)
#         time_step = eval_env.step(action_step.action)
#         video.append_data(eval_py_env.render())
#   return embed_mp4(filename)

# create_policy_eval_video(agent.policy, "trained-agent")

# create_policy_eval_video(random_policy, "random-agent")

