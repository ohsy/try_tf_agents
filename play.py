"""
play the game with tf-agents
2025.1
Sangyeop Oh

variables in __main__ are global, 
but I used those like locals to explain the relations
except config and logger.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import json
import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import argparse
import pathlib 
import numpy as np
import base64
import matplotlib
import matplotlib.pyplot as plt

import reverb
import mujoco
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.agents.ddpg import ddpg_agent, actor_network, critic_network
from tf_agents.agents.td3 import td3_agent
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.agents.cql import cql_sac_agent
from tf_agents.environments import suite_gym, tf_py_environment, wrappers
# from tf_agents.environments import suite_pybullet
from tf_agents.networks import sequential, actor_distribution_network, q_network, categorical_q_network

from tf_agents.drivers import py_driver, dynamic_step_driver, dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics, py_metrics
from tf_agents.policies import actor_policy, greedy_policy, py_tf_eager_policy, random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils, tf_uniform_replay_buffer
from tf_agents.trajectories import Trajectory
from tf_agents.specs import tensor_spec
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.utils import common, tensor_normalizer
from tf_agents.train.utils import spec_utils, train_utils
from tf_agents.system import system_multiprocessing as multiprocessing

from env_daiso.env_for_tfa import DaisoSokcho
from game import Game, compute_avg_return, collect_trajectory
from game_multiagent import MultiAgentGame, compute_avg_return_for_multiagent, collect_trajectory_for_multiagent

# to suppress warning of data_manager.py
import warnings


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

def getLogger(filepath="./main.log", log_level_name="INFO"):
    logger = logging.getLogger("game")
    logging.basicConfig(
            level = logging.getLevelName(log_level_name),
            # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            format = "%(asctime)s - %(levelname)s - %(message)s",
            handlers = [ 
                # logging.FileHandler(filename=filepath, mode="w"),
                RotatingFileHandler(filename=filepath, maxBytes=100000, backupCount=10),
                logging.StreamHandler(sys.stdout)
            ]
    )
    return logger


def get_tf_env_specs(tf_env, py_env):
    tf_observation_spec = tf_train_env.observation_spec()
    tf_action_spec = tf_train_env.action_spec()
    tf_env_step_spec = tf_train_env.time_step_spec()
    logger.debug(f"py_observation_spec: {py_env.observation_spec()}")
    logger.debug(f"py_action_spec: {py_env.action_spec()}")
    logger.debug(f"py_env_step_spec: {py_env.time_step_spec()}")
    logger.info(f"tf_observation_spec: {tf_observation_spec}")
    logger.info(f"tf_action_spec: {tf_action_spec}")
    logger.info(f"tf_env_step_spec: {tf_env_step_spec}")
    return tf_observation_spec, tf_action_spec, tf_env_step_spec


def get_tf_agent_specs_for_multiagent(agents, tf_action_spec):
    tf_agent_collect_data_spec = Trajectory(
        agents[0].collect_data_spec.step_type,
        agents[0].collect_data_spec.observation,
        tf_action_spec,
        agents[0].collect_data_spec.policy_info,
        agents[0].collect_data_spec.next_step_type,
        agents[0].collect_data_spec.reward,
        agents[0].collect_data_spec.discount)
    logger.info(f"tf_agent_collect_data_spec: {tf_agent_collect_data_spec}")
    return tf_agent_collect_data_spec


def get_env(envName, envWrapper, num_actions):
    if envName in ['CartPole-v0','Pendulum-v1','Reacher-v2']:
        py_train_env = suite_gym.load(envName)
        py_eval_env = suite_gym.load(envName)
    elif envName in ['MinitaurBulletEnv-v0']:
        py_train_env = suite_pybullet.load(envName)
        py_eval_env = suite_pybullet.load(envName)
    elif envName in ['DaisoSokcho']:
        py_train_env = DaisoSokcho()
        py_eval_env = DaisoSokcho()
    elif envName in ['Pendulum-v1_discrete','Reacher-v2_discrete']:
        eName = envName.split('_discrete')[0]
        py_train_env = wrappers.ActionDiscretizeWrapper(suite_gym.load(eName), num_actions=num_actions)
        py_eval_env = wrappers.ActionDiscretizeWrapper(suite_gym.load(eName), num_actions=num_actions)
    elif envName in ['DaisoSokcho_discrete']: 
        py_train_env = wrappers.ActionDiscretizeWrapper(DaisoSokcho(), num_actions=num_actions)
        py_eval_env = wrappers.ActionDiscretizeWrapper(DaisoSokcho(), num_actions=num_actions)
    elif envName in ['DaisoSokcho_discrete_unit1']:
        py_train_env = DaisoSokcho()
        _num_actions = [int(n) for n in (py_train_env.action_spec().maximum - py_train_env.action_spec().minimum) + 1]
        logger.info(f"in get_env(), for {envName}, num_actions={_num_actions}")
        py_train_env = wrappers.ActionDiscretizeWrapper(py_train_env, num_actions=_num_actions)
        py_eval_env = wrappers.ActionDiscretizeWrapper(DaisoSokcho(), num_actions=_num_actions)
    else:
        sys.exit(f"environment {envName} is not supported.")

    if envWrapper in ['history']:
        py_train_env = wrappers.HistoryWrapper(py_train_env)
        py_eval_env = wrappers.HistoryWrapper(py_eval_env)

    tf_train_env = tf_py_environment.TFPyEnvironment(py_train_env)
    tf_eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)

    return py_train_env, py_eval_env, tf_train_env, tf_eval_env


def get_agent(agentName, tf_env_step_spec, tf_observation_spec, tf_action_spec, epsilon_greedy):
    qnet_fc_layer_params = config['qnet_fc_layer_params']
    actor_fc_layer_params = config['actor_fc_layer_params']
    critic_observation_fc_layer_params = config['critic_observation_fc_layer_params']
    critic_action_fc_layer_params = config['critic_action_fc_layer_params']
    critic_joint_fc_layer_params = config['critic_joint_fc_layer_params']

    # for CDQN
    num_atoms = config['num_atoms']
    if "CartPole" in envName:
        min_q_value = 0
        max_q_value = 200
    elif "Pendulum" in envName:
        min_q_value = -1500
        max_q_value = 0
    elif "Reacher" in envName:
        min_q_value = -60
        max_q_value = 0
    elif "DaisoSokcho" in envName:
        min_q_value = -600
        max_q_value = 0
    else:
        min_q_value = config['min_q_value']
        max_q_value = config['max_q_value']
    logger.info(f"min_q_value={min_q_value}, max_q_value={max_q_value}")
    n_step_update = config['n_step_update']

    learning_rate = config['learning_rate']
    actor_learning_rate = config['actor_learning_rate']
    critic_learning_rate = config['critic_learning_rate']
    alpha_learning_rate = config['alpha_learning_rate']
    target_update_tau = config['target_update_tau']
    target_update_period = config['target_update_period']
    gamma = config['gamma']
    reward_scale_factor = config['reward_scale_factor']
    agent = None

    if agentName in ["BC"]:
        # TEMP: DQN is used
        q_net = q_network.QNetwork(
            tf_observation_spec,
            tf_action_spec,
            fc_layer_params=qnet_fc_layer_params)
        agent = behavioral_cloning_agent.BehavioralCloningAgent(
            tf_env_step_spec,
            tf_action_spec,
            cloning_network=q_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            # loss_fn=common.element_wise_squared_loss,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=tf.Variable(0, dtype=tf.int64))

    elif agentName in ["CQL_SAC"]:
        # cf. https://github.com/tensorflow/agents/blob/master/tf_agents/agents/cql/cql_sac_agent.py
        # actor_fc_layer_params = [256, 256]
        # critic_joint_fc_layer_params = [256, 256, 256]
        # Agent params
        # batch_size: int = 256
        bc_steps: int = 0
        actor_learning_rate: types.Float = 3e-5
        critic_learning_rate: types.Float = 3e-4
        alpha_learning_rate: types.Float = 3e-4
        reward_scale_factor: types.Float = 1.0
        cql_alpha_learning_rate: types.Float = 3e-4
        cql_alpha: types.Float = 5.0
        cql_tau: types.Float = 10.0
        num_cql_samples: int = 10
        reward_noise_variance: Union[types.Float, tf.Variable] = 0.0
        include_critic_entropy_term: bool = False
        use_lagrange_cql_alpha: bool = True
        log_cql_alpha_clipping: Optional[Tuple[types.Float, types.Float]] = None
        softmax_temperature: types.Float = 1.0
        # Data and Reverb Replay Buffer params
        reward_shift: types.Float = 0.0
        action_clipping: Optional[Tuple[types.Float, types.Float]] = None
        data_shuffle_buffer_size: int = 100
        data_prefetch: int = 10
        data_take: Optional[int] = None
        pad_end_of_episodes: bool = False
        reverb_port: Optional[int] = None
        min_rate_limiter: int = 1
        # Others
        policy_save_interval: int = 10000
        eval_interval: int = 10000
        summary_interval: int = 1000
        learner_iterations_per_call: int = 1
        eval_episodes: int = 10
        debug_summaries: bool = False
        summarize_grads_and_vars: bool = False
        seed: Optional[int] = None
        
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            tf_observation_spec, 
            tf_action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))
        critic_net = critic_network.CriticNetwork(
            (tf_observation_spec, tf_action_spec),
            observation_fc_layer_params=critic_observation_fc_layer_params,
            action_fc_layer_params=critic_action_fc_layer_params,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')
        agent = cql_sac_agent.CqlSacAgent(
            tf_env_step_spec,
            tf_action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate),
            alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_learning_rate),
            cql_alpha=cql_alpha,
            num_cql_samples=num_cql_samples,
            include_critic_entropy_term=include_critic_entropy_term,
            use_lagrange_cql_alpha=use_lagrange_cql_alpha,
            cql_alpha_learning_rate=cql_alpha_learning_rate,
            target_update_tau=5e-3,
            target_update_period=1,
            random_seed=seed,
            cql_tau=cql_tau,
            reward_noise_variance=reward_noise_variance,
            num_bc_steps=bc_steps,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=0.99,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=None,
            log_cql_alpha_clipping=log_cql_alpha_clipping,
            softmax_temperature=softmax_temperature,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_utils.create_train_step(),)

    elif agentName in ["DQN"]:
        q_net = q_network.QNetwork(
            tf_observation_spec,
            tf_action_spec,
            fc_layer_params=qnet_fc_layer_params)
        agent = dqn_agent.DqnAgent(
            tf_env_step_spec,
            tf_action_spec,
            q_network=q_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            epsilon_greedy=epsilon_greedy,
            td_errors_loss_fn=common.element_wise_squared_loss,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=tf.Variable(0, dtype=tf.int64))

    elif agentName in ["CDQN"]:
        categorical_q_net = categorical_q_network.CategoricalQNetwork(
            tf_observation_spec,
            tf_action_spec,
            num_atoms=num_atoms,
            fc_layer_params=qnet_fc_layer_params)
        agent = categorical_dqn_agent.CategoricalDqnAgent(
            tf_env_step_spec,
            tf_action_spec,
            categorical_q_network=categorical_q_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            epsilon_greedy=epsilon_greedy,
            min_q_value=min_q_value,
            max_q_value=max_q_value,
            n_step_update=n_step_update,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=tf.Variable(0, dtype=tf.int64))

    elif agentName in ["DDPG"]:
        actor_net = actor_network.ActorNetwork(
            tf_observation_spec, 
            tf_action_spec,
            fc_layer_params=actor_fc_layer_params)
        critic_net = critic_network.CriticNetwork(
            (tf_observation_spec, tf_action_spec),
            observation_fc_layer_params=critic_observation_fc_layer_params,
            action_fc_layer_params=critic_action_fc_layer_params,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')
        agent = ddpg_agent.DdpgAgent(
            tf_env_step_spec,
            tf_action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=train_utils.create_train_step())

    elif agentName in ["TD3"]:
        actor_net = actor_network.ActorNetwork(
            tf_observation_spec, 
            tf_action_spec,
            fc_layer_params=actor_fc_layer_params)
        critic_net = critic_network.CriticNetwork(
            (tf_observation_spec, tf_action_spec),
            observation_fc_layer_params=critic_observation_fc_layer_params,
            action_fc_layer_params=critic_action_fc_layer_params,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')
        agent = td3_agent.Td3Agent(
            tf_env_step_spec,
            tf_action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=train_utils.create_train_step())

    elif agentName in ["SAC"]:
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            tf_observation_spec, 
            tf_action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))
        critic_net = critic_network.CriticNetwork(
            (tf_observation_spec, tf_action_spec),
            observation_fc_layer_params=critic_observation_fc_layer_params,
            action_fc_layer_params=critic_action_fc_layer_params,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')
        agent = sac_agent.SacAgent(
            tf_env_step_spec,
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

    if agent is not None:
        agent.initialize()
    return agent


def get_agents(agentName, tf_env_step_spec, tf_observation_spec, merged_tf_action_spec, epsilon_greedy):
    if merged_tf_action_spec.maximum.ndim == 0:  # spec.maximum is np.ndarray
        maxima = np.full(merged_tf_action_spec.shape, merged_tf_action_spec.maximum)
    else:
        maxima = merged_tf_action_spec.maximum
    logger.info(f"multiagent: tf_action_spec maxima={maxima}")

    single_agentName = agentName.split('_multiagent')[0]
    agents = []
    for ix, mx in enumerate(maxima):
        tf_action_spec = tensor_spec.from_spec(
            BoundedArraySpec(
                shape=(),
                dtype=np.int32, # CartPole's action_spec dtype =int64. ActionDiscreteWrapper(DaisoSokcho)'s =int32
                name=f'action{ix}', 
                minimum=0, 
                maximum=mx))
        agent = get_agent(single_agentName, tf_env_step_spec, tf_observation_spec, tf_action_spec, epsilon_greedy)
        agents.append(agent)
        agents[ix].initialize()
    logger.info(f"multiagent: number of agents = {len(agents)}")
    return agents


def get_replay_buffer(agentName, tf_train_env, agent_collect_data_spec, reverb_port, isPerUsed):
    batch_size = config['batch_size']
    replay_buffer_max_length = config['replay_buffer_max_length']
    # NOTE: num_frames = max_length * env.batch_size and default env.batch_size = 1". capacity is max num_frames.
    table_name = 'uniform_table'

    if replay_bufferName in ['tf_uniform']:
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent_collect_data_spec,
            batch_size=tf_train_env.batch_size,  # adding batch_size, not sampling
            max_length=replay_buffer_max_length)
        observers=[replay_buffer.add_batch]
    elif replay_bufferName in ['reverb']:
        replay_buffer_signature = tensor_spec.from_spec(agent_collect_data_spec)
        logger.info(f"replay_buffer_signature={replay_buffer_signature}")
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
        logger.info(f"after adding outer dim, replay_buffer_signature={replay_buffer_signature}")

        table = reverb.Table(
            table_name,
            max_size=replay_buffer_max_length,
            # sampler=(reverb.selectors.Prioritized() if isPerUsed else reverb.selectors.Uniform()),
            sampler=(reverb.libpybind.PrioritizedSelector(priority_exponent=0.8) if isPerUsed else reverb.selectors.Uniform()),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature)

        if checkpointPath_toRestore is not None and reverb_checkpointPath_toRestore is not None:  # restore from checkpoint
            logger.info(f"restoring reverb replay_buffer from reverb_checkpointPath_toRestore={reverb_checkpointPath_toRestore}")
            reverb_checkpointer = reverb.checkpointers.DefaultCheckpointer(path=reverb_checkpointPath_toRestore)
            reverb_server = reverb.Server([table], checkpointer=reverb_checkpointer, port=reverb_port)
        else:
            reverb_server = reverb.Server([table], port=reverb_port)
        # reverb_client = reverb.Client(f"localhost:{reverb_port}")

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            agent_collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server)

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2,
            stride_length=1)

        observers = [rb_observer]

    if agentName in ['BC']: # All of the Tensors in `value` must have one outer dimensions: must have shape `[B] + spec.shape`.
        dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=batch_size)
                # num_steps=1
                # single_deterministic_pass=True)
    elif agentName in ['CQL_SAC']: # All of the Tensors in `value` must have two outer dimensions: must have shape `[B, T] + spec.shape`. 
        dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=batch_size,
                num_steps=2)
                # single_deterministic_pass=True)
    else:
        dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=batch_size,   # adds outer dim like [64,...]
                num_steps=2                     # adds outer dim like [64, 2, ...]
                ).prefetch(3)
    iterator = iter(dataset)

    logger.info(f"ending get_replay_buffer(), replay_buffer.num_frames()={replay_buffer.num_frames()}")
    return replay_buffer, dataset, iterator, observers


def fill_replay_buffer(driverName, py_train_env, tf_train_env, policy, replay_buffer, observers, num_env_steps_to_collect_init):
    """
    collect env steps with policy and save to the replay buffer.
    """
    before_fill = time.time()
    logger.info(f"replay_buffer.capacity={replay_buffer.capacity}")
    logger.info(f"starting fill_replay_buffer(), replay_buffer.num_frames()={replay_buffer.num_frames()}")
    replay_buffer.clear()
    logger.info(f"before filling, replay_buffer.num_frames()={replay_buffer.num_frames()}")
    # NOTE: num_frames = max_length * env.batch_size and default env.batch_size = 1". capacity is max num_frames.
    num_env_steps_to_collect_per_time_step = config['num_env_steps_to_collect_per_time_step']
    num_episodes_to_collect_init = config['num_episodes_to_collect_init']
    num_episodes_to_collect_per_time_step = config['num_episodes_to_collect_per_time_step']

    env_step = py_train_env.reset()
    if driverName in ['py']:
        init_driver = py_driver.PyDriver(
            py_train_env,
            py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True),
            observers,
            max_steps=num_env_steps_to_collect_init)
        init_driver.run(env_step) 
    elif driverName in ['dynamic_step']:
        init_driver = dynamic_step_driver.DynamicStepDriver(
            tf_train_env,
            policy,
            observers=observers,
            num_steps=num_env_steps_to_collect_per_time_step)
        for _ in range(num_env_steps_to_collect_init):
            init_driver.run()
    elif driverName in ['dynamic_episode']:
        init_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_train_env,
            policy,
            observers=observers,
            num_episodes=num_episodes_to_collect_per_time_step,)
        for _ in range(num_episodes_to_collect_init):
            init_driver.run()
    elif driverName in ['none']:
        tf_train_env.reset()
        for _ in range(num_env_steps_to_collect_init):
            collect_trajectory(logger, tf_train_env, replay_buffer, policy)

    logger.info(f"after filling with random_policies or agent.collect_policy, replay_buffer.num_frames()={replay_buffer.num_frames()}")
    logger.info(f"filling time = {time.time() - before_fill:.3f}")


def fill_replay_buffer_for_multiagent(tf_train_env, tf_random_policies, replay_buffer, num_env_steps_to_collect_init):
    before_fill = time.time()
    logger.info(f"replay_buffer.capacity={replay_buffer.capacity}")
    logger.info(f"before filling, replay_buffer.num_frames()={replay_buffer.num_frames()}")
    # NOTE: num_frames = max_length * env.batch_size and default env.batch_size = 1". capacity is max num_frames.
    tf_train_env.reset()
    # collect env steps with random policy and save to the replay buffer.
    for _ in range(num_env_steps_to_collect_init):
        collect_trajectory_for_multiagent(logger, tf_train_env, replay_buffer, policies=tf_random_policies)
    logger.info(f"after filling with random_policies, replay_buffer.num_frames()={replay_buffer.num_frames()}")
    logger.info(f"filling time = {time.time() - before_fill:.3f}")


def get_driver(py_train_env, tf_train_env, agent_collect_policy, observers):
    num_env_steps_to_collect_per_time_step = config['num_env_steps_to_collect_per_time_step']
    num_episodes_to_collect_per_time_step = config['num_episodes_to_collect_per_time_step']

    if driverName in ['py']:
        driver = py_driver.PyDriver(
            py_train_env,
            py_tf_eager_policy.PyTFEagerPolicy(agent_collect_policy, use_tf_function=True),
            observers,
            max_steps=num_env_steps_to_collect_per_time_step)
    elif driverName in ['dynamic_step']:
        driver = dynamic_step_driver.DynamicStepDriver(
            tf_train_env,
            agent_collect_policy,
            observers=observers,
            num_steps=num_env_steps_to_collect_per_time_step)
    elif driverName in ['dynamic_episode']:
        # environment_steps_metric = tf_metrics.EnvironmentSteps()
        # step_metrics = [tf_metrics.NumberOfEpisodes(), environment_steps_metric,]
        # train_metrics = step_metrics + [
        #     tf_metrics.AverageReturnMetric(batch_size=tf_train_env.batch_size),
        #     tf_metrics.AverageEpisodeLengthMetric(batch_size=tf_train_env.batch_size),]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_train_env,
            agent_collect_policy,
            observers=observers, # observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=num_episodes_to_collect_per_time_step,)
    elif driverName in ['none']:
        driver = None
    else:
        driver = None
    return driver


def restore_agent_and_replay_buffer(checkpointPath_toRestore, reverb_checkpointPath_toRestore, agent, replay_buffer=None, fill_after_restore=False):
    """
    restore agent and replay buffer if checkpoints are given
    reverb replay_buffer is restored elsewhere
    """
    before_restore = time.time()
    if replay_buffer is not None:
        logger.info(f"replay_buffer.capacity={replay_buffer.capacity}")
        logger.info(f"before restoring with checkpointer, replay_buffer.num_frames()={replay_buffer.num_frames()}")
    logger.info(f"restoring from checkpointPath_toRestore={checkpointPath_toRestore}")

    if checkpointPath_toRestore is None:
        return
    kwargs = {'agent': agent, 'policy': agent.policy, 'global_step': agent.train_step_counter}
    if reverb_checkpointPath_toRestore is None and replay_buffer is not None and not fill_after_restore:  
        kwargs['replay_buffer'] = replay_buffer
    checkpointer = common.Checkpointer(ckpt_dir=checkpointPath_toRestore, **kwargs)
    checkpointer.initialize_or_restore()

    if replay_buffer is not None:
        logger.info(f"after restoring with checkpointer, replay_buffer.num_frames()={replay_buffer.num_frames()}")
    logger.info(f"restoring time = {time.time() - before_restore:.3f}")


def restore_agent_and_replay_buffer_for_multiagent(checkpointPath_toRestore, reverb_checkpointPath_toRestore, agents, replay_buffer):
    """
    restore agent and replay buffer if checkpoints are given
    reverb replay_buffer is restored elsewhere
    """
    if checkpointPath_toRestore is None:
        return
    for ix, agent in enumerate(agents):
        ckptPath = os.path.join(checkpointPath_toRestore, f'{ix}')
        if ix == 0: 
            restore_agent_and_replay_buffer(ckptPath, reverb_checkpointPath_toRestore, agent, replay_buffer, fill_after_restore)
        else:
            restore_agent_and_replay_buffer(ckptPath, reverb_checkpointPath_toRestore, agent)  # excluding replay_buffer


def game_run_multiagent(agentName, tf_train_env, tf_observation_spec, tf_action_spec, epsilon_greedy, 
        checkpointPath_toRestore, reverb_checkpointPath_toRestore, checkpointPath_toSave, checkpoint_max_to_keep, replay_bufferName, reverb_port, isPerUsed):

    agents = get_agents(agentName, tf_env_step_spec, tf_observation_spec, tf_action_spec, epsilon_greedy)  # list of agents 
    tf_agent_collect_data_spec = get_tf_agent_specs_for_multiagent(agents, tf_action_spec)

    replay_buffer, dataset, iterator, observers = get_replay_buffer(agentName, tf_train_env, tf_agent_collect_data_spec, reverb_port, isPerUsed)

    checkpointers_toSave = []
    for ix, agent in enumerate(agents):
        checkpointPath = os.path.join(checkpointPath_toSave, f'{ix}')
        kwargs = {'agent': agent, 'policy': agent.policy, 'global_step': agent.train_step_counter}
        if replay_bufferName != 'reverb' and ix == 0:  # since there is only one replay_buffer for multi-agents
            kwargs['replay_buffer'] = replay_buffer
        checkpointer = common.Checkpointer(ckpt_dir=checkpointPath, max_to_keep=checkpoint_max_to_keep, **kwargs)
        checkpointers_toSave.append(checkpointer)
    reverb_client_toSave = reverb.Client(f"localhost:{reverb_port}") if replay_buffer.__class__.__name__ in ['ReverbReplayBuffer'] else None

    tf_random_policy = random_tf_policy.RandomTFPolicy(tf_env_step_spec, agents[0].action_spec) 
    # tf_random_policies = [random_tf_policy.RandomTFPolicy(tf_env_step_spec, ag.action_spec) for ag in agents]
    tf_random_policies = [tf_random_policy for ag in agents]
    random_return = compute_avg_return_for_multiagent(tf_eval_env, policies=tf_random_policies)
    logger.info(f"random_policy avg_return={random_return:.3f}")

    if checkpointPath_toRestore is None:
        fill_replay_buffer_for_multiagent(tf_train_env, tf_random_policies, replay_buffer, num_env_steps_to_collect_init)
    else:
        restore_agent_and_replay_buffer_for_multiagent(checkpointPath_toRestore, reverb_checkpointPath_toRestore, agents, replay_buffer, fill_after_restore)
        if fill_after_restore == 'true':
            fill_replay_buffer_for_multiagent(tf_train_env, agent.collect_policy, replay_buffer, num_env_steps_to_collect_init)

    game = MultiAgentGame(config)
    if config['isSummaryWriterUsed']:
        with summaryWriter.as_default():
            game.run(logger, tf_train_env, tf_eval_env, agents, replay_buffer, iterator, checkpointers_toSave, reverb_client_toSave)
    else:
        game.run(logger, tf_train_env, tf_eval_env, agents, replay_buffer, iterator, checkpointers_toSave, reverb_client_toSave)



if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    # Keep using keras-2 (tf-keras) rather than keras-3 (keras).
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    np.set_printoptions(precision=6, threshold=sys.maxsize, linewidth=160, suppress=True)
    #   if (not tf.test.is_built_with_cuda()) or len(tf.config.list_physical_devices('GPU')) == 0:
    #       os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with open(os.getcwd()+'/config.json') as f:
        config = json.load(f)

    if config['isGpuUsed']:
        gpus = tf.config.list_physical_devices('GPU')
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            sys.exit(f"tf.config.experimental.set_memory_growth() is not working for {gpus}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f"online arguments={sys.argv}", flush=True)
    parser = argparse.ArgumentParser(description="argpars parser used")
    parser.add_argument('-e', '--environment', type=str, 
            choices=['CartPole-v0','Pendulum-v1','Pendulum-v1_discrete','Reacher-v2','Reacher-v2_discrete',\
                    'DaisoSokcho','DaisoSokcho_discrete','DaisoSokcho_discrete_unit1'])
    parser.add_argument('-w', '--environment_wrapper', type=str, choices=['history'], 
            help="environment wrapper: 'history' adds observation and action history to the environment's observations.")
    parser.add_argument('-a', '--agent', type=str, choices=['DQN','DQN_multiagent','CDQN','CDQN_multiagent','DDPG','TD3','SAC','BC','CQL_SAC'])
    parser.add_argument('-r', '--replay_buffer', type=str, choices=['reverb','tf_uniform'], help="'reverb' must be used with driver 'py'")
    parser.add_argument('-d', '--driver', type=str, choices=['py','dynamic_step','dynamic_episode','none']) 
    parser.add_argument('-c', '--checkpoint_path', type=str, help="to restore")
    parser.add_argument('-f', '--fill_after_restore', type=str, help="fill replay_buffer with agent.policy after restoring agent", 
            choices=['true','false'])
    parser.add_argument('-p', '--reverb_checkpoint_path', type=str, help="to restore: parent directory of saved path," + 
            " which is output when saved, like '/tmp/tmp6j63a_f_' of '/tmp/tmp6j63a_f_/2024-10-27T05:22:20.16401174+00:00'")
    parser.add_argument('-n', '--num_actions', type=int, help="number of actions for ActionDiscretizeWrapper")
    parser.add_argument('-i', '--num_env_steps_to_collect_init', type=int, help="number of initial collect steps")
    parser.add_argument('-g', '--epsilon_greedy', type=float, help="epsilon for epsilon_greedy")
    parser.add_argument('-o', '--reverb_port', type=int, help="reverb port for reverb.Client and Server")
    args = parser.parse_args()

    envName = config["environment"] if args.environment is None else args.environment
    envWrapper = config["environment_wrapper"] if args.environment_wrapper is None else args.environment_wrapper
    agentName = config["agent"] if args.agent is None else args.agent
    replay_bufferName = config["replay_buffer"] if args.replay_buffer is None else args.replay_buffer
    driverName = config["driver"] if args.driver is None else args.driver
    checkpointPath_toRestore = args.checkpoint_path
    fill_after_restore = args.fill_after_restore
    reverb_checkpointPath_toRestore = args.reverb_checkpoint_path
    num_actions = config['num_actions'] if args.num_actions is None else args.num_actions
    num_env_steps_to_collect_init = config['num_env_steps_to_collect_init'] if args.num_env_steps_to_collect_init is None else args.num_env_steps_to_collect_init
    epsilon_greedy = config['epsilon_greedy'] if args.epsilon_greedy is None else args.epsilon_greedy
    reverb_port = config['reverb_port'] if args.reverb_port is None else args.reverb_port
    if replay_bufferName == 'reverb':
        assert driverName == 'py', "replay_buffer 'reverb' must be used with driver 'py'"
    if agentName in ['BC','CQL_SAC']:
        assert not (checkpointPath_toRestore is None and reverb_checkpointPath_toRestore is None), \
                "checkpoint_path or reverb_checkpoint_path must not be None for agent BC or CQL_SAC"

    date_time = datetime.now().strftime('%m%d_%H%M%S')
    resultPath = f"{config['resultPath']}/{envName}_{agentName}_{date_time}"
    logPath = f"{resultPath}/log/game.log"
    pathlib.Path(logPath).parent.mkdir(exist_ok=True, parents=True)
    logger = getLogger(filepath=logPath, log_level_name=config["log_level_name"])
    summaryPath = f"{resultPath}/log/summary"  # directory 
    summaryWriter = tf.summary.create_file_writer(summaryPath)
    checkpointPath_toSave = f'{resultPath}/model'
    checkpoint_max_to_keep = config['checkpoint_max_to_keep']
    isPerUsed = config['isPerUsed']

    logger.info(f"config={config}")
    logger.info(f"args={args}")
    logger.info(f"environment={envName}")  
    logger.info(f"envWrapper={envWrapper}")  
    logger.info(f"agent={agentName}")
    logger.info(f"replay_buffer={replay_bufferName}")
    logger.info(f"driver={driverName}")
    if 'discrete' in envName:
        logger.info(f"ActionDiscretizeWrapper num_actions={num_actions}")
    logger.info(f"num_env_steps_to_collect_init={num_env_steps_to_collect_init}")
    logger.info(f"epsilon_greedy={epsilon_greedy}")
    logger.info(f"checkpointPath_toRestore={checkpointPath_toRestore}")
    logger.info(f"checkpointPath_toSave={checkpointPath_toSave}")
    logger.info(f"reverb_port={reverb_port}")


    py_train_env, py_eval_env, tf_train_env, tf_eval_env = get_env(envName, envWrapper, num_actions)
    tf_observation_spec, tf_action_spec, tf_env_step_spec = get_tf_env_specs(tf_train_env, py_train_env)

    # epsilon decay cf. https://github.com/tensorflow/agents/blob/master/tf_agents/agents/categorical_dqn/examples/train_eval_atari.py
    # global_step = tf.compat.v1.train.get_or_create_global_step()
    # epsilon_greedy=0.01
    # epsilon_decay_period=10000  # 1000000: period over which to decay epsilon, from 1.0 to epsilon_greedy
    # epsilon = tf.compat.v1.train.polynomial_decay(1.0, global_step, epsilon_decay_period, end_learning_rate=epsilon_greedy)


    if 'multiagent' in agentName:
        #NOTE: game_run_multiagent() can be used actually since variables in __main__ can be seen in functions
        game_run_multiagent(agentName, tf_train_env, tf_observation_spec, tf_action_spec, epsilon_greedy, 
                checkpointPath_toRestore, reverb_checkpointPath_toRestore, checkpointPath_toSave, checkpoint_max_to_keep, replay_bufferName, reverb_port, isPerUsed)
        sys.exit(0)


    agent = get_agent(agentName, tf_env_step_spec, tf_observation_spec, tf_action_spec, epsilon_greedy)  # one agent 
    tf_agent_collect_data_spec = agent.collect_data_spec 
    logger.info(f"tf_agent_collect_data_spec: {tf_agent_collect_data_spec}")

    replay_buffer, dataset, iterator, observers = get_replay_buffer(agentName, tf_train_env, tf_agent_collect_data_spec, reverb_port, isPerUsed)

    kwargs = {'agent': agent, 'policy': agent.policy, 'global_step': agent.train_step_counter}
    if replay_bufferName != 'reverb':
        kwargs['replay_buffer'] = replay_buffer
    checkpointer_toSave = common.Checkpointer(ckpt_dir=checkpointPath_toSave, max_to_keep=checkpoint_max_to_keep, **kwargs)
    # reverb_client = reverb.Client(f"localhost:{reverb_port}") if replay_buffer.__class__.__name__ in ['ReverbReplayBuffer'] else None
    reverb_client_toSave = reverb.Client(f"localhost:{reverb_port}") if replay_bufferName == 'reverb' else None

    tf_random_policy = random_tf_policy.RandomTFPolicy(tf_env_step_spec, tf_action_spec)
    random_return = compute_avg_return(tf_eval_env, tf_random_policy)
    logger.info(f"random_policy avg_return={random_return:.3f}")

    if checkpointPath_toRestore is None:
        fill_replay_buffer(driverName, py_train_env, tf_train_env, tf_random_policy, replay_buffer, observers, num_env_steps_to_collect_init)
    else:
        restore_agent_and_replay_buffer(checkpointPath_toRestore, reverb_checkpointPath_toRestore, agent, replay_buffer, fill_after_restore)
        if fill_after_restore == 'true':
            fill_replay_buffer(driverName, py_train_env, tf_train_env, agent.collect_policy, replay_buffer, observers, num_env_steps_to_collect_init)

    driver = get_driver(py_train_env, tf_train_env, agent.collect_policy, observers)

    game = Game(config, isTrainOnly=True) if agentName in ['BC','CQL_SAC'] \
            else Game(config)

    if config['isSummaryWriterUsed']:
        with summaryWriter.as_default():
            game.run(logger, py_train_env, tf_eval_env, agent, replay_buffer, iterator, driver, checkpointer_toSave, reverb_client_toSave)
    else:
        game.run(logger, py_train_env, tf_eval_env, agent, replay_buffer, iterator, driver, checkpointer_toSave, reverb_client_toSave)

