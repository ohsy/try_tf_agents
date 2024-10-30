
from __future__ import absolute_import, division, print_function

import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import reverb
import time
from datetime import datetime

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.environments import suite_gym, tf_py_environment
# from tf_agents.environments import suite_pybullet
from tf_agents.networks import sequential, actor_distribution_network, value_network

from tf_agents.drivers import py_driver, dynamic_step_driver, dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics, py_metrics
from tf_agents.policies import actor_policy, greedy_policy, py_tf_eager_policy, random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils, tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common, tensor_normalizer
from tf_agents.train.utils import spec_utils, train_utils

from env_daiso.env_for_tfa import DaisoSokcho

# to suppress warning of data_manager.py
import warnings
warnings.filterwarnings("ignore")


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics
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

def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

class NormalizedActorPolicy(actor_policy.ActorPolicy):
    def __init__(self, *args, **kwargs):
        super(NormalizedActorPolicy, self).__init__(
            *args,
            # observation_normalizer=tensor_normalizer.EMATensorNormalizer(tf_observation_spec),
            observation_normalizer=tensor_normalizer.StreamingTensorNormalizer(tf_observation_spec),
            **kwargs
        )

is_gpu_used = True
if is_gpu_used:
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        sys.exit(f"tf.config.experimental.set_memory_growth() is not working for {gpus}")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# env_name = "CartPole-v0" # @param {type:"string"}
env_name = "Pendulum-v1" # @param {type:"string"}
# env_name = "MinitaurBulletEnv-v0" # @param {type:"string"}
# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
# env_name = "DaisoSokcho" # @param {type:"string"}
# agent_name = "dqn"
# agent_name = "cdqn"
agent_name = "ppo"
# agent_name = "sac"
# replay_buffer_name = "tf_uniform"
replay_buffer_name = "reverb"
driver_name = "py"
# driver_name = "dynamic_step"
# driver_name = "dynamic_episode""
save_or_restore = "save"
# save_or_restore = "restore"

num_iterations = 500 # 20000 # @param {type:"integer"}
initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1# @param {type:"integer"}
replay_buffer_max_length = 500  # @param {type:"integer"}

batch_size = 8  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"integer"}
critic_learning_rate = 6e-4  # @param {type:"number"}
actor_learning_rate = 3e-4  # @param {type:"number"}
alpha_learning_rate = 3e-4  # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

fc_layer_params = (100, 50)
actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)
value_fc_layer_params = (256, 256)

# for PPO
num_parallel_environments = 30
if agent_name in ["ppo"]:
    batch_size = num_parallel_environments
num_environment_steps = 20000  # 25000000
num_epochs = 25
collect_episodes_per_iteration = num_parallel_environments

log_interval = 20 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 100 # @param {type:"integer"}
# policy_save_interval = 5000 # @param {type:"integer"}

if env_name in ['CartPole-v0', 'Pendulum-v1']:
    py_train_env = suite_gym.load(env_name)
    py_eval_env = suite_gym.load(env_name)
elif env_name in ['MinitaurBulletEnv-v0']:
    py_train_env = suite_pybullet.load(env_name)
    py_eval_env = suite_pybullet.load(env_name)
elif env_name in ['DaisoSokcho']:
    py_train_env = DaisoSokcho()
    py_eval_env = DaisoSokcho()
else:
    sys.exit(f"env_name {env_name} is not supported.")


print(f"observation spec: {py_train_env.observation_spec()}")
print(f"action spec: {py_train_env.action_spec()}")

tf_train_env = tf_py_environment.TFPyEnvironment(py_train_env)
tf_eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)
tf_observation_spec = tf_train_env.observation_spec()
tf_action_spec = tf_train_env.action_spec()
tf_time_step_spec = tf_train_env.time_step_spec()
print(f"tf observation spec: {tf_observation_spec}")
print(f"tf action spec: {tf_action_spec}")
print(f"tf time_step Spec: {tf_time_step_spec}", flush=True)


if agent_name in ["dqn", 'cdqn']:

    # action_tensor_spec = tensor_spec.from_spec(py_train_env.action_spec())
    # num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    num_actions = tf_action_spec.maximum - tf_action_spec.minimum + 1

    if agent_name in ["dqn"]:
        dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        q_net = sequential.Sequential(dense_layers + [q_values_layer])
        agent = dqn_agent.DqnAgent(
            tf_time_step_spec,
            tf_action_spec,
            q_network=q_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            td_errors_loss_fn=common.element_wise_squared_loss,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=tf.Variable(0, dtype=tf.int64))

    if agent_name in ["cdqn"]:
        categorical_q_net = categorical_q_network.CategoricalQNetwork(
            tf_observation_spec,
            tf_action_spec,
            num_atoms=num_atoms,
            fc_layer_params=fc_layer_params)
        agent = categorical_dqn_agent.CategoricalDqnAgent(
            tf_time_step_spec,
            tf_action_spec,
            categorical_q_network=categorical_q_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            min_q_value=min_q_value,
            max_q_value=max_q_value,
            n_step_update=n_step_update,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=tf.Variable(0, dtype=tf.int64))

elif agent_name in ["ppo"]:
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_observation_spec,
        tf_action_spec,
        fc_layer_params=actor_fc_layer_params,
        activation_fn=tf.keras.activations.tanh)
    value_net = value_network.ValueNetwork(
        tf_observation_spec,
        fc_layer_params=value_fc_layer_params,
        activation_fn=tf.keras.activations.tanh)
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
        debug_summaries=True,
        summarize_grads_and_vars=True,
        train_step_counter=train_utils.create_train_step())  # todo

elif agent_name in ["sac"]:
    critic_net = critic_network.CriticNetwork(
        (tf_observation_spec, tf_action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_observation_spec, 
        tf_action_spec,
        fc_layer_params=actor_fc_layer_params,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork))
    agent = sac_agent.SacAgent(
        tf_time_step_spec,
        tf_action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_learning_rate),
        actor_policy_ctor=NormalizedActorPolicy,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        debug_summaries=True,
        summarize_grads_and_vars=True,
        train_step_counter=train_utils.create_train_step())

agent.initialize()


table_name = 'uniform_table'

if replay_buffer_name in ['tf_uniform']:
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_train_env.batch_size,
        max_length=replay_buffer_max_length)
    observers=[replay_buffer.add_batch]

elif replay_buffer_name in ['reverb']:

    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    print(f"replay_buffer_signature={replay_buffer_signature}", flush=True)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
    print(f"replay_buffer_signature={replay_buffer_signature}", flush=True)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    # reverb_server = reverb.Server([table])
    if save_or_restore in ['save']:
        reverb_server = reverb.Server([table], port=8000)
        reverb_client = reverb.Client('localhost:8000')
    elif save_or_restore in ['restore']:
        # reverb_checkpoint_path = "/tmp/tmp6j63a_f_/2024-10-27T05:22:20.16401174+00:00"
        reverb_checkpoint_path = "/tmp/tmp6j63a_f_"  # parent directory of the returned path of save
        reverb_checkpointer = reverb.checkpointers.DefaultCheckpointer(path=reverb_checkpoint_path)
        reverb_server = reverb.Server(tables=[table], checkpointer=reverb_checkpointer, port=8000)
        reverb_client = reverb.Client('localhost:8000')

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

    observers = [rb_observer]

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
iterator = iter(dataset)


if save_or_restore in ['restore']:
    checkpoint_path = os.path.join(os.path.abspath(os.getcwd()), f'result/checkpoint_1027_0514')
    if replay_buffer_name in ['tf_uniform']:
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_path,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter)
    elif replay_buffer_name in ['reverb']:
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_path,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            # replay_buffer=replay_buffer,
            global_step=agent.train_step_counter)

    train_checkpointer.initialize_or_restore()


before_all = time.time()

tf_random_policy = random_tf_policy.RandomTFPolicy(tf_time_step_spec, tf_action_spec)
random_return = compute_avg_return(tf_eval_env, tf_random_policy, num_eval_episodes)
print(f"random_return = {random_return}", flush=True)

print(f"before initial driver run")
print(f"dataset.cardinality()={dataset.cardinality()}", flush=True)
print(f"replay_buffer.capacity={replay_buffer.capacity}", flush=True)
print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)

time_step = py_train_env.reset()

if driver_name in ['py']:
    init_collect_driver = py_driver.PyDriver(
        py_train_env,
        py_tf_eager_policy.PyTFEagerPolicy(tf_random_policy, use_tf_function=True),
        observers,
        max_steps=initial_collect_steps)

    init_collect_driver.run(time_step) 

    collect_driver = py_driver.PyDriver(
        py_train_env,
        py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
        observers,
        max_steps=collect_steps_per_iteration)
elif driver_name in ['dynamic_step']:
    init_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_train_env,
        tf_random_policy,
        observers=observers,
        num_steps=collect_steps_per_iteration)

    for _ in range(initial_collect_steps):
        init_collect_driver.run()

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_train_env,
        agent.collect_policy,
        observers=observers,
        num_steps=collect_steps_per_iteration)
elif driver_name in ['dynamic_episode']:
    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [tf_metrics.NumberOfEpisodes(), environment_steps_metric,]
    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(batch_size=batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=batch_size),]
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration,)


# for _ in range(initial_collect_steps // 83):
#     reset_time_step=py_train_env.reset()
#     print(f"after env.reset, time_step={reset_time_step}", flush=True)
#     for _ in range(83):
#         init_collect_driver.run()

print(f"after initial driver run")
print(f"dataset.cardinality()={dataset.cardinality()}", flush=True)
print(f"replay_buffer.capacity={replay_buffer.capacity}", flush=True)
print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(tf.Variable(0, dtype=tf.int64))

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(tf_eval_env, agent.policy, num_eval_episodes)
print(f"before for-loop, avg_return={avg_return}", flush=True)
returns = [avg_return]

date_time = datetime.now().strftime('%m%d_%H%M%S')
summary_path = os.path.join(os.path.abspath(os.getcwd()), f'log/summary_{date_time}')
summary_writer = tf.summary.create_file_writer(summary_path)

time_step = py_train_env.reset()

before = time.time()
for _ in range(num_iterations):

    # Collect a few steps and save to the replay buffer.
    if driver_name in ['py']:
        time_step, _ = collect_driver.run(time_step)
    elif driver_name in ['dynamic_step','dynamic_episode']:
        collect_driver.run()

    if agent_name in ['ppo']:
        trajectories = replay_buffer.gather_all()
        train_loss = agent.train(experience=trajectories).loss
        replay_buffer.clear()
    else:
        # experience, unused_info = next(iterator)
        experience, unused_info = iterator.get_next()  # experience as tensor
        # print(f"experience={experience}", flush=True)
        with summary_writer.as_default():
            loss_info = agent.train(experience)
            train_loss = loss_info.loss

    train_step = agent.train_step_counter.numpy()

    if train_step % log_interval == 0:
        after = time.time()
        print(f'train_step = {train_step}: loss = {train_loss}, time used = {after - before}', flush=True)
        before = after

    if train_step % eval_interval == 0:
        avg_return = compute_avg_return(tf_eval_env, agent.policy, num_eval_episodes)
        print(f'train_step = {train_step}: average return = {avg_return}', flush=True)
        returns.append(avg_return)

after_all = time.time()
print(f"total time = {after_all-before_all}")


checkpoint_path = os.path.join(os.path.abspath(os.getcwd()), f'result/checkpoint_{date_time}')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_path,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=agent.train_step_counter
)
train_checkpointer.save(agent.train_step_counter)

if replay_buffer_name in ['reverb']:
    reverb_checkpoint_path = reverb_client.checkpoint()
    print(f"reverb_checkpoint_path={reverb_checkpoint_path}", flush=True)


