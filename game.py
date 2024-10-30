"""
game with tf-agents
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
from tf_agents.system import system_multiprocessing as multiprocessing


from env_daiso.env_for_tfa import DaisoSokcho

# to suppress warning of data_manager.py
import warnings


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

def getLogger(filepath="./log.log"):
    logger = logging.getLogger("game")
    logger.setLevel(logging.INFO) #   INFO, DEBUG
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fileHandler = logging.FileHandler(filename=filepath, mode="w")
    fileHandler.setLevel(logging.INFO) # INFO, DEBUG
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger




class Game:
    def __init__(self, config, checkpointPath):
        self.config = config
        self.checkpointPath = checkpointPath
        self.num_train_steps_to_save_model = config["num_train_steps_to_save_model"]

    def run(self, py_train_env, tf_eval_env, agent, replay_buffer, iterator, init_driver, driver):

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        agent.train = common.function(agent.train)

        # Reset the train step.
        agent.train_step_counter.assign(tf.Variable(0, dtype=tf.int64))

        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(tf_eval_env, agent.policy, num_episodes_to_eval)
        print(f"before for-loop, avg_return={avg_return}", flush=True)
        returns = [avg_return]

        time_step = py_train_env.reset()
        before = time.time()
        for _ in range(num_train_steps):

            # Collect a few steps and save to the replay buffer.
            if driverName in ['py']:
                time_step, _ = driver.run(time_step)
            elif driverName in ['dynamic_step','dynamic_episode']:
                driver.run()

            if agentName in ['PPOClip']:
                trajectories = replay_buffer.gather_all()
                train_loss = agent.train(experience=trajectories).loss
                replay_buffer.clear()
            else:
                # experience, unused_info = next(iterator)
                experience, unused_info = iterator.get_next()  # experience as tensor
                # print(f"experience={experience}", flush=True)
                loss_info = agent.train(experience)
                train_loss = loss_info.loss

            train_step = agent.train_step_counter.numpy()

            if train_step % num_train_steps_to_log == 0:
                after = time.time()
                print(f'train_step = {train_step}: loss = {train_loss:.3f}, time = {after-before:.3f}', flush=True)
                before = after

            if train_step % num_train_steps_to_eval == 0:
                avg_return = compute_avg_return(tf_eval_env, agent.policy, num_episodes_to_eval)
                print(f'train_step = {train_step}: average return = {avg_return:.3f}', flush=True)
                returns.append(avg_return)

        after_all = time.time()
        print(f"total time = {after_all-before_all:.3f}")


        # checkpointPath = os.path.join(os.path.abspath(os.getcwd()), f'{self.resultPath}/model')
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpointPath,
            max_to_keep=2,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter)
        train_checkpointer.save(agent.train_step_counter)

        if replay_bufferName in ['reverb']:
            reverb_client = reverb.Client(f"localhost:{config['reverb_port']}")
            reverb_checkpointPath = reverb_client.checkpoint()
            # print(f"reverb_checkpointPath={reverb_checkpointPath}", flush=True)



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

    num_train_steps = config['num_train_steps']
    num_train_steps_to_log = config['num_train_steps_to_log']
    num_train_steps_to_eval = config['num_train_steps_to_eval']
    num_train_steps_to_save_model = config['num_train_steps_to_save_model']
    num_episodes_to_eval = config['num_episodes_to_eval']

    qnet_fc_layer_params = config['qnet_fc_layer_params']
    actor_fc_layer_params = config['actor_fc_layer_params']
    critic_observation_fc_layer_params = config['critic_observation_fc_layer_params']
    critic_action_fc_layer_params = config['critic_action_fc_layer_params']
    critic_joint_fc_layer_params = config['critic_joint_fc_layer_params']
    value_fc_layer_params = config['value_fc_layer_params']

    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    actor_learning_rate = config['actor_learning_rate']
    critic_learning_rate = config['critic_learning_rate']
    alpha_learning_rate = config['alpha_learning_rate']
    target_update_tau = config['target_update_tau']
    target_update_period = config['target_update_period']
    gamma = config['gamma']
    reward_scale_factor = config['reward_scale_factor']

    replay_buffer_max_length = config['replay_buffer_max_length']
    # NOTE: num_frames = capacity = max_length * env.batch_size and default env.batch_size = 1" : None 
    num_initial_collect_steps = config['num_initial_collect_steps']
    num_collect_steps_per_train_step = config['num_collect_steps_per_train_step']

    # for PPO
    num_parallel_envs = config['num_parallel_envs']
    num_env_steps = config['num_env_steps']
    num_epochs = config['num_epochs']
    num_collect_episodes_per_train_step = config['num_collect_episodes_per_train_step']

    parser = argparse.ArgumentParser(description="argpars parser used")
    parser.add_argument('-e', '--environment', type=str, choices=['CartPole-v0','Pendulum-v1','DaisoSokcho'])
    parser.add_argument('-a', '--agent', type=str, choices=['DQN','CDQN','PPOClip','SAC'])
    parser.add_argument('-r', '--replay_buffer', type=str, choices=['reverb','tf_uniform'])
    parser.add_argument('-d', '--driver', type=str, choices=['py','dynamic_step','dynamic_episode']) 
    parser.add_argument('-c', '--checkpoint_path', type=str)
    parser.add_argument('-p', '--reverb_checkpoint_path', type=str, help="parent directory of save path, which is output when saved, like '/tmp/tmp6j63a_f_' of '/tmp/tmp6j63a_f_/2024-10-27T05:22:20.16401174+00:00'")
    args = parser.parse_args()
    args = parser.parse_args()

    envName = config["environment"] if args.environment is None else args.environment
    agentName = config["agent"] if args.agent is None else args.agent
    replay_bufferName = config["replay_buffer"] if args.replay_buffer is None else args.replay_buffer
    driverName = config["driver"] if args.driver is None else args.driver
    checkpointPath = args.checkpoint_path
    reverb_checkpointPath = args.reverb_checkpoint_path

    date_time = datetime.now().strftime('%m%d_%H%M%S')
    resultPath = f"{config['resultPath']}/{envName}_{agentName}_{date_time}"
    logPath = f"{resultPath}/log/game.log"
    pathlib.Path(logPath).parent.mkdir(exist_ok=True, parents=True)
    logger = getLogger(filepath = logPath)
    summaryPath = f"{resultPath}/log/summary"  # directory 
    summaryWriter = tf.summary.create_file_writer(summaryPath)
    checkpointPath = f'{resultPath}/model'


    if envName in ['CartPole-v0', 'Pendulum-v1']:
        py_train_env = suite_gym.load(envName)
        py_eval_env = suite_gym.load(envName)
    elif envName in ['MinitaurBulletEnv-v0']:
        py_train_env = suite_pybullet.load(envName)
        py_eval_env = suite_pybullet.load(envName)
    elif envName in ['DaisoSokcho']:
        py_train_env = DaisoSokcho()
        py_eval_env = DaisoSokcho()
    else:
        sys.exit(f"environment {envName} is not supported.")

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


    # action_tensor_spec = tensor_spec.from_spec(py_train_env.action_spec())
    # num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    if agentName in ["DQN"]:
        num_actions = tf_action_spec.maximum - tf_action_spec.minimum + 1
        dense_layers = [dense_layer(num_units) for num_units in qnet_fc_layer_params]
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

    if agentName in ["CDQN"]:
        num_actions = tf_action_spec.maximum - tf_action_spec.minimum + 1
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

    elif agentName in ["PPOClip"]:
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


    table_name = 'uniform_table'

    if replay_bufferName in ['tf_uniform']:
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=tf_train_env.batch_size,  # adding batch_size, not sampling
            max_length=replay_buffer_max_length)
        observers=[replay_buffer.add_batch]

    elif replay_bufferName in ['reverb']:

        replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
        print(f"replay_buffer_signature={replay_buffer_signature}", flush=True)
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
        print(f"after adding outer dim, replay_buffer_signature={replay_buffer_signature}", flush=True)

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


    # restore agent and replay buffer if checkpoints are given
    if checkpointPath is not None and reverb_checkpointPath is not None:  # reverb replay_buffer is restored elsewhere
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpointPath,
            max_to_keep=2,
            agent=agent,
            policy=agent.policy,
            # replay_buffer=replay_buffer,
            global_step=agent.train_step_counter)
        train_checkpointer.initialize_or_restore()
    elif checkpointPath is not None and reverb_checkpointPath is None:
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpointPath,
            max_to_keep=2,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter)
        train_checkpointer.initialize_or_restore()


    before_all = time.time()

    tf_random_policy = random_tf_policy.RandomTFPolicy(tf_time_step_spec, tf_action_spec)
    random_return = compute_avg_return(tf_eval_env, tf_random_policy, num_episodes_to_eval)
    print(f"random_return = {random_return}", flush=True)

    print(f"before initial driver run")
    print(f"dataset.cardinality()={dataset.cardinality()}", flush=True)
    print(f"replay_buffer.capacity={replay_buffer.capacity}", flush=True)
    print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)

    time_step = py_train_env.reset()

    if driverName in ['py']:
        init_driver = py_driver.PyDriver(
            py_train_env,
            py_tf_eager_policy.PyTFEagerPolicy(tf_random_policy, use_tf_function=True),
            observers,
            max_steps=num_initial_collect_steps)

        init_driver.run(time_step) 

        driver = py_driver.PyDriver(
            py_train_env,
            py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
            observers,
            max_steps=num_collect_steps_per_train_step)
    elif driverName in ['dynamic_step']:
        init_driver = dynamic_step_driver.DynamicStepDriver(
            tf_train_env,
            tf_random_policy,
            observers=observers,
            num_steps=num_collect_steps_per_train_step)

        for _ in range(num_initial_collect_steps):
            init_driver.run()

        driver = dynamic_step_driver.DynamicStepDriver(
            tf_train_env,
            agent.collect_policy,
            observers=observers,
            num_steps=num_collect_steps_per_train_step)
    elif driverName in ['dynamic_episode']:
        environment_steps_metric = tf_metrics.EnvironmentSteps()
        step_metrics = [tf_metrics.NumberOfEpisodes(), environment_steps_metric,]
        train_metrics = step_metrics + [
            tf_metrics.AverageReturnMetric(batch_size=batch_size),
            tf_metrics.AverageEpisodeLengthMetric(batch_size=batch_size),]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_train_env,
            agent.collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=num_collect_episodes_per_train_step,)

    print(f"after initial driver run")
    print(f"dataset.cardinality()={dataset.cardinality()}", flush=True)
    print(f"replay_buffer.capacity={replay_buffer.capacity}", flush=True)
    print(f"replay_buffer.num_frames()={replay_buffer.num_frames()}", flush=True)


    logger.info(f"environment = {envName}")  
    # logger.info(f"env name: {env.unwrapped.spec.id}")  # like "CartPole-v0"
    logger.info(f"agent = {agentName}")
    logger.info(f"observation spec: {tf_observation_spec}")
    logger.info(f"action spec: {tf_action_spec}")
    logger.info(f"time_step spec: {tf_time_step_spec}")
    logger.info(f"config={config}")


    game = Game(config, checkpointPath)
    with summaryWriter.as_default():
        if agentName in ["PPOClip"]:
            multiprocessing.handle_main(functools.partial(app.run, game.run))
        else:
            game.run(py_train_env, tf_eval_env, agent, replay_buffer, iterator, init_driver, driver)

