"""
play the game with tf-agents
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import json
import time
from datetime import datetime
import logging
import argparse
import pathlib 
import numpy as np
import base64
import matplotlib
import matplotlib.pyplot as plt

import reverb
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.environments import suite_gym, tf_py_environment, wrappers
# from tf_agents.environments import suite_pybullet
from tf_agents.networks import sequential, actor_distribution_network, q_network, categorical_q_network

from tf_agents.drivers import py_driver, dynamic_step_driver, dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics, py_metrics
from tf_agents.policies import actor_policy, greedy_policy, py_tf_eager_policy, random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils, tf_uniform_replay_buffer
from tf_agents.trajectories import Trajectory
from tf_agents.specs import tensor_spec
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.utils import common, tensor_normalizer
from tf_agents.train.utils import spec_utils, train_utils
from tf_agents.system import system_multiprocessing as multiprocessing

from env_daiso.env_for_tfa import DaisoSokcho
from game import Game, compute_avg_return
from game_multiagent import MultiAgentGame, multiagent_compute_avg_return, collect_trajectory

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
                    logging.FileHandler(filename=filepath, mode="w"),
                    logging.StreamHandler(sys.stdout)
            ]
    )
    return logger


def get_tf_env_specs(logger, tf_env, py_env):
    tf_observation_spec = tf_train_env.observation_spec()
    tf_action_spec = tf_train_env.action_spec()
    tf_time_step_spec = tf_train_env.time_step_spec()
    logger.debug(f"py_observation_spec: {py_env.observation_spec()}")
    logger.debug(f"py_action_spec: {py_env.action_spec()}")
    logger.debug(f"py_time_step_spec: {py_env.time_step_spec()}")
    logger.info(f"tf_observation_spec: {tf_observation_spec}")
    logger.info(f"tf_action_spec: {tf_action_spec}")
    logger.info(f"tf_time_step_spec: {tf_time_step_spec}")
    return tf_observation_spec, tf_action_spec, tf_time_step_spec


def get_tf_agent_specs(logger, agent, tf_action_spec):
    if type(agent) is not list: 
        tf_agent_collect_data_spec = agent.collect_data_spec 
    else:
        tf_agent_collect_data_spec = Trajectory(
            agent[0].collect_data_spec.step_type,
            agent[0].collect_data_spec.observation,
            tf_action_spec,
            agent[0].collect_data_spec.policy_info,
            agent[0].collect_data_spec.next_step_type,
            agent[0].collect_data_spec.reward,
            agent[0].collect_data_spec.discount)
    logger.info(f"tf_agent_collect_data_spec: {tf_agent_collect_data_spec}")
    return tf_agent_collect_data_spec


def get_env(config, envName, envWrapper):
    if envName in ['CartPole-v0', 'Pendulum-v1']:
        py_train_env = suite_gym.load(envName)
        py_eval_env = suite_gym.load(envName)
    elif envName in ['MinitaurBulletEnv-v0']:
        py_train_env = suite_pybullet.load(envName)
        py_eval_env = suite_pybullet.load(envName)
    elif envName in ['DaisoSokcho']:
        py_train_env = DaisoSokcho()
        py_eval_env = DaisoSokcho()
    elif envName in ['Pendulum-v1_discrete']:
        eName = envName.split('_discrete')[0]
        py_train_env = wrappers.ActionDiscretizeWrapper(suite_gym.load(eName), num_actions=5)
        py_eval_env = wrappers.ActionDiscretizeWrapper(suite_gym.load(eName), num_actions=5)
    elif envName in ['DaisoSokcho_discrete']:
        py_train_env = wrappers.ActionDiscretizeWrapper(DaisoSokcho(), num_actions=(6,8,6,6,6))
        py_eval_env = wrappers.ActionDiscretizeWrapper(DaisoSokcho(), num_actions=(6,8,6,6,6))
    else:
        sys.exit(f"environment {envName} is not supported.")

    if envWrapper in ['history']:
        py_train_env = wrappers.HistoryWrapper(py_train_env)
        py_eval_env = wrappers.HistoryWrapper(py_eval_env)

    tf_train_env = tf_py_environment.TFPyEnvironment(py_train_env)
    tf_eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)

    return py_train_env, py_eval_env, tf_train_env, tf_eval_env


def get_agent(config, tf_observation_spec, tf_action_spec):
    qnet_fc_layer_params = config['qnet_fc_layer_params']
    actor_fc_layer_params = config['actor_fc_layer_params']
    critic_observation_fc_layer_params = config['critic_observation_fc_layer_params']
    critic_action_fc_layer_params = config['critic_action_fc_layer_params']
    critic_joint_fc_layer_params = config['critic_joint_fc_layer_params']

    # for CDQN
    num_atoms = config['num_atoms']
    min_q_value = config['min_q_value']
    max_q_value = config['max_q_value']
    n_step_update = config['n_step_update']

    learning_rate = config['learning_rate']
    actor_learning_rate = config['actor_learning_rate']
    critic_learning_rate = config['critic_learning_rate']
    alpha_learning_rate = config['alpha_learning_rate']
    target_update_tau = config['target_update_tau']
    target_update_period = config['target_update_period']
    gamma = config['gamma']
    reward_scale_factor = config['reward_scale_factor']

    if agentName in ["DQN"]:
        q_net = q_network.QNetwork(
            tf_observation_spec,
            tf_action_spec,
            fc_layer_params=qnet_fc_layer_params)
        agent = dqn_agent.DqnAgent(
            tf_time_step_spec,
            tf_action_spec,
            q_network=q_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            td_errors_loss_fn=common.element_wise_squared_loss,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            train_step_counter=tf.Variable(0, dtype=tf.int64))
        agent.initialize()
        return agent

    if agentName in ["DQN_multiagent"]:
        tf_action_specs = []
        q_nets = []
        agents = []
        for ix, mx in enumerate(tf_action_spec.maximum):
            tf_action_specs.append(
                tensor_spec.from_spec(
                    BoundedArraySpec(
                        shape=(),
                        dtype=np.int32, # CartPole's action_spec dtype =int64. ActionDiscreteWrapper(DaisoSokcho)'s =int32
                        name=f'action{ix}', 
                        minimum=0, 
                        maximum=mx)))
            q_nets.append(
                q_network.QNetwork(
                    tf_observation_spec,
                    tf_action_specs[ix],
                    fc_layer_params=qnet_fc_layer_params))
            agents.append(
                dqn_agent.DqnAgent(
                    tf_time_step_spec,
                    tf_action_specs[ix],
                    q_network=q_nets[ix],
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    td_errors_loss_fn=common.element_wise_squared_loss,
                    debug_summaries=True,
                    summarize_grads_and_vars=True,
                    train_step_counter=tf.Variable(0, dtype=tf.int64)))
            agents[ix].initialize()
        return agents

    if agentName in ["CDQN"]:
        categorical_q_net = categorical_q_network.CategoricalQNetwork(
            tf_observation_spec,
            tf_action_spec,
            num_atoms=num_atoms,
            fc_layer_params=qnet_fc_layer_params)
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
        agent.initialize()
        return agent

    elif agentName in ["SAC"]:
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
        return agent


def get_replay_buffer(config, logger, tf_train_env, agent_collect_data_spec):
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
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature)

        if checkpointPath is not None and reverb_checkpointPath is not None:  # restore from checkpoint
            reverb_checkpointer = reverb.checkpointers.DefaultCheckpointer(path=reverb_checkpointPath)
            reverb_server = reverb.Server([table], checkpointer=reverb_checkpointer, port=config['reverb_port'])
        else:
            reverb_server = reverb.Server([table], port=config['reverb_port'])
        # reverb_client = reverb.Client(f"localhost:{config['reverb_port']}")

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

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    return replay_buffer, iterator, observers


def fill_replay_buffer(config, py_train_env, tf_train_env, tf_random_policy, observers):
    num_init_collect_steps = config['num_init_collect_steps']
    num_collect_steps_per_train_step = config['num_collect_steps_per_train_step']
    num_init_collect_episodes = config['num_init_collect_episodes']
    num_collect_episodes_per_train_step = config['num_collect_episodes_per_train_step']

    time_step = py_train_env.reset()
    if driverName in ['py']:
        init_driver = py_driver.PyDriver(
            py_train_env,
            py_tf_eager_policy.PyTFEagerPolicy(tf_random_policy, use_tf_function=True),
            observers,
            max_steps=num_init_collect_steps)
        init_driver.run(time_step) 
    elif driverName in ['dynamic_step']:
        init_driver = dynamic_step_driver.DynamicStepDriver(
            tf_train_env,
            tf_random_policy,
            observers=observers,
            num_steps=num_collect_steps_per_train_step)
        for _ in range(num_init_collect_steps):
            init_driver.run()
    elif driverName in ['dynamic_episode']:
        init_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_train_env,
            tf_random_policy,
            observers=observers,
            num_episodes=num_collect_episodes_per_train_step,)
        for _ in range(num_init_collect_episodes):
            init_driver.run()


def fill_replay_buffer_for_multiagent(config, tf_train_env, tf_random_policies, replay_buffer):
    num_init_collect_steps = config['num_init_collect_steps']

    tf_train_env.reset()
    if 'multiagent' in agentName:
        # Collect random policy steps and save to the replay buffer.
        for _ in range(self.num_init_collect_steps):
            collect_trajectory(logger, tf_train_env, replay_buffer, policies=tf_random_policies)


def get_driver(config, py_train_env, tf_train_env, agent_collect_policy, observers):
    num_init_collect_steps = config['num_init_collect_steps']
    num_collect_steps_per_train_step = config['num_collect_steps_per_train_step']
    num_collect_episodes_per_train_step = config['num_collect_episodes_per_train_step']

    if driverName in ['py']:
        driver = py_driver.PyDriver(
            py_train_env,
            py_tf_eager_policy.PyTFEagerPolicy(agent_collect_policy, use_tf_function=True),
            observers,
            max_steps=num_collect_steps_per_train_step)
    elif driverName in ['dynamic_step']:
        driver = dynamic_step_driver.DynamicStepDriver(
            tf_train_env,
            agent_collect_policy,
            observers=observers,
            num_steps=num_collect_steps_per_train_step)
    elif driverName in ['dynamic_episode']:
        # environment_steps_metric = tf_metrics.EnvironmentSteps()
        # step_metrics = [tf_metrics.NumberOfEpisodes(), environment_steps_metric,]
        # train_metrics = step_metrics + [
        #     tf_metrics.AverageReturnMetric(batch_size=tf_train_env.batch_size),
        #     tf_metrics.AverageEpisodeLengthMetric(batch_size=tf_train_env.batch_size),]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_train_env,
            agent_collect_policy,
            # observers=[replay_buffer.add_batch] + train_metrics,
            observers=observers,
            num_episodes=num_collect_episodes_per_train_step,)
    return driver


def restore_agent_and_replay_buffer(checkpointPath, reverb_checkpointPath, agent, replay_buffer):
    # restore agent and replay buffer if checkpoints are given
    if checkpointPath is None:
        return
    if reverb_checkpointPath is not None:  # reverb replay_buffer is restored elsewhere
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpointPath,
            max_to_keep=2,
            agent=agent,
            policy=agent.policy,
            # replay_buffer=replay_buffer,
            global_step=agent.train_step_counter)
    elif reverb_checkpointPath is None:
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpointPath,
            max_to_keep=2,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter)
    train_checkpointer.initialize_or_restore()

def restore_agent_and_replay_buffer_for_multiagent(checkpointPath, reverb_checkpointPath, agents, replay_buffer):
    # restore agents and replay buffer if checkpoints are given
    if checkpointPath is None:
        return
    for ix, agent in enumerate(agents):
        ckptPath = os.path.join(checkpointPath, f'{ix}')
        if (reverb_checkpointPath is not None) or (reverb_checkpointPath is None and ix != 0):
            train_checkpointer = common.Checkpointer(
                ckpt_dir=ckptPath,
                max_to_keep=2,
                agent=agent,
                policy=agent.policy,
                global_step=agent.train_step_counter)
        elif reverb_checkpointPath is None and ix == 0:
            train_checkpointer = common.Checkpointer(
                ckpt_dir=ckptPath,
                max_to_keep=2,
                agent=agent,
                policy=agent.policy,
                replay_buffer=replay_buffer,
                global_step=agent.train_step_counter)
        train_checkpointer.initialize_or_restore()



if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    # Keep using keras-2 (tf-keras) rather than keras-3 (keras).
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    np.set_printoptions(precision=6, threshold=sys.maxsize, linewidth=160, suppress=True)
    #   if (not tf.test.is_built_with_cuda()) or len(tf.config.list_physical_devices('GPU')) == 0:
    #       os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with open(os.getcwd()+'/config.json') as f:
        config = json.load(f)

    if config['isGPUUsed']:
        gpus = tf.config.list_physical_devices('GPU')
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            sys.exit(f"tf.config.experimental.set_memory_growth() is not working for {gpus}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser(description="argpars parser used")
    parser.add_argument('-e', '--environment', type=str, 
            choices=['CartPole-v0','Pendulum-v1','Pendulum-v1_discrete','DaisoSokcho','DaisoSokcho_discrete'])
    parser.add_argument('-w', '--environment_wrapper', type=str, choices=['history'], help="environment wrapper")
    parser.add_argument('-a', '--agent', type=str, choices=['DQN','DQN_multiagent','CDQN','SAC'])
    parser.add_argument('-r', '--replay_buffer', type=str, choices=['reverb','tf_uniform'])
    parser.add_argument('-d', '--driver', type=str, choices=['py','dynamic_step','dynamic_episode']) 
    parser.add_argument('-c', '--checkpoint_path', type=str, help="to restore")
    parser.add_argument('-p', '--reverb_checkpoint_path', type=str, help="to restore: parent directory of saved path, which is output when saved, like '/tmp/tmp6j63a_f_' of '/tmp/tmp6j63a_f_/2024-10-27T05:22:20.16401174+00:00'")
    args = parser.parse_args()
    args = parser.parse_args()

    envName = config["environment"] if args.environment is None else args.environment
    envWrapper = config["environment_wrapper"] if args.environment_wrapper is None else args.environment_wrapper
    agentName = config["agent"] if args.agent is None else args.agent
    replay_bufferName = config["replay_buffer"] if args.replay_buffer is None else args.replay_buffer
    driverName = config["driver"] if args.driver is None else args.driver
    checkpointPath = args.checkpoint_path
    reverb_checkpointPath = args.reverb_checkpoint_path

    date_time = datetime.now().strftime('%m%d_%H%M%S')
    resultPath = f"{config['resultPath']}/{envName}_{agentName}_{date_time}"
    logPath = f"{resultPath}/log/game.log"
    pathlib.Path(logPath).parent.mkdir(exist_ok=True, parents=True)
    logger = getLogger(filepath=logPath, log_level_name=config["log_level_name"])
    summaryPath = f"{resultPath}/log/summary"  # directory 
    summaryWriter = tf.summary.create_file_writer(summaryPath)
    checkpointPath_toSave = f'{resultPath}/model'
    logger.info(f"config={config}")
    logger.info(f"args={args}")
    logger.info(f"environment = {envName}")  
    logger.info(f"envWrapper = {envWrapper}")  
    logger.info(f"agent = {agentName}")

    py_train_env, py_eval_env, tf_train_env, tf_eval_env = get_env(config, envName, envWrapper)
    tf_observation_spec, tf_action_spec, tf_time_step_spec = get_tf_env_specs(logger, tf_train_env, py_train_env)

    agent = get_agent(config, tf_observation_spec, tf_action_spec)  # one agent or list of agents if multiagent
    tf_agent_collect_data_spec = get_tf_agent_specs(logger, agent, tf_action_spec)

    replay_buffer, iterator, observers = get_replay_buffer(config, logger, tf_train_env, tf_agent_collect_data_spec)

    if 'multiagent' in agentName:
        agents = agent  # list
        tf_random_policies = [random_tf_policy.RandomTFPolicy(tf_time_step_spec, ag.action_spec) for ag in agents]
        random_return = multiagent_compute_avg_return(tf_eval_env, policies=tf_random_policies)
        logger.info(f"random_policy return = {random_return}")
        logger.info(f"replay_buffer.capacity={replay_buffer.capacity}")
        logger.info(f"before filling or restoring, replay_buffer.num_frames()={replay_buffer.num_frames()}, (may have already restored for reverb)")
        if checkpointPath is None:
            fill_replay_buffer_for_multiagent(config, tf_train_env, tf_random_policies, replay_buffer)
            logger.info(f"after filling, replay_buffer.num_frames()={replay_buffer.num_frames()}")
        else:
            restore_agent_and_replay_buffer_for_multiagent(checkpointPath, reverb_checkpointPath, agent, replay_buffer)
            logger.info(f"after restoring, replay_buffer.num_frames()={replay_buffer.num_frames()}")

        game = MultiAgentGame(config, checkpointPath_toSave)
        with summaryWriter.as_default():
            game.run(logger, tf_train_env, tf_eval_env, agents, replay_buffer, iterator)
    else: 
        tf_random_policy = random_tf_policy.RandomTFPolicy(tf_time_step_spec, tf_action_spec)
        random_return = compute_avg_return(tf_eval_env, tf_random_policy)
        logger.info(f"random_policy return = {random_return}")
        logger.info(f"replay_buffer.capacity={replay_buffer.capacity}")
        logger.info(f"before filling or restoring, replay_buffer.num_frames()={replay_buffer.num_frames()}, (may have already restored for reverb)")
        # NOTE: num_frames = max_length * env.batch_size and default env.batch_size = 1". capacity is max num_frames.
        if checkpointPath is None:
            fill_replay_buffer(config, py_train_env, tf_train_env, tf_random_policy, observers)
            logger.info(f"after filling, replay_buffer.num_frames()={replay_buffer.num_frames()}")
        else:
            restore_agent_and_replay_buffer(checkpointPath, reverb_checkpointPath, agent, replay_buffer)
            logger.info(f"after restoring, replay_buffer.num_frames()={replay_buffer.num_frames()}")

        driver = get_driver(config, py_train_env, tf_train_env, agent.collect_policy, observers)

        game = Game(config, checkpointPath_toSave)
        with summaryWriter.as_default():
            game.run(logger, py_train_env, tf_eval_env, agent, replay_buffer, iterator, driver)

