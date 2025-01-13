"""
game with tf-agents
"""

import os
import time
import numpy as np
# import reverb
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.trajectories import Trajectory, from_transition, PolicyStep
from tf_agents.policies import random_tf_policy


def collect_trajectory(logger, environment, replay_buffer, policies=None, agents=None):
    """
    based on the following code for one agent:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)
    """
    assert not(policies is None and agents is None), f"either policies or agents must not be None"
    if policies is None:
        policies = []
        for agent in agents:
            policies.append(agent.collect_policy)

    env_step = environment.current_time_step()
    # action_step = policy.action(env_step)
    actions = []
    for policy in policies:
        action_step = policy.action(env_step)
        actions.append(action_step.action)
    merged_action = merge_action(actions)
    # logger.debug(f"action_step={action_step}")
    action_step = PolicyStep(action=merged_action)
    next_env_step = environment.step(merged_action)
    # logger.debug(f"action_step={action_step}")
    traj = from_transition(env_step, action_step, next_env_step)
    replay_buffer.add_batch(traj)


def merge_action(actions):
    """
    actions: list of actions
    """
    # print(f"in merge_action, actions={actions}",flush=True)
    merged = tf.reshape(actions, [1,len(actions)])
    # merged = tf.cast(merged, dtype=tf.int32)
    # print(f"in merge_action, merged={merged}",flush=True)
    return merged
    # return np.array([actions], dtype=np.int32)
    # return [actions]


def trajectories_from_merged_trajectory(merged, num_trajectories):
    trajectories = []
    for ix in range(num_trajectories):
        trajectories.append(
            Trajectory(
                merged.step_type,
                merged.observation,
                merged.action[:, :, ix],
                merged.policy_info,
                merged.next_step_type,
                merged.reward,
                merged.discount))
    return trajectories


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics
def multiagent_compute_avg_return(environment, policies=None, agents=None, num_episodes=10):
    assert not(policies is None and agents is None), f"either policies or agents must not be None"
    if policies is None:
        policies = []
        for agent in agents:
            policies.append(agent.policy)

    total_return = 0.0
    for _ in range(num_episodes):

        env_step = environment.reset()
        episode_return = 0.0

        while not env_step.is_last():
            actions = []
            for policy in policies:
                action_step = policy.action(env_step)
                actions.append(action_step.action) 
            merged_action = merge_action(actions)
            env_step = environment.step(merged_action)
            episode_return += env_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


class MultiAgentGame:
    def __init__(self, config, checkpointPath_toSave):
        # self.num_init_collect_steps = config['num_init_collect_steps']
        self.num_collect_steps_per_train_step = config['num_collect_steps_per_train_step']
        # self.num_episodes_to_eval = config['num_episodes_to_eval']
        self.checkpointPath_toSave = checkpointPath_toSave
        self.reverb_port = config['reverb_port']
        # self.num_train_steps = config['num_train_steps']
        # self.num_train_steps_to_log = max((int) (self.num_train_steps / 250), 1)
        # self.num_train_steps_to_eval = max((int) (self.num_train_steps / 50), 1)
        # self.num_train_steps_to_save = max((int) (self.num_train_steps / 5), 1)
        self.num_time_steps = config['num_time_steps']
        self.num_time_steps_to_log = max((int) (self.num_time_steps / 500), 1)
        self.num_time_steps_to_eval = max((int) (self.num_time_steps / 50), 1)
        self.num_time_steps_to_train = config['num_time_steps_to_train']
        self.num_train_steps_to_save = config['num_train_steps_to_save']
        self.num_episodes_to_eval = config['num_episodes_to_eval']
        self.train_step = 0

    def save(self, logger, agents, replay_buffer, checkpoint_max_to_keep):
        # checkpointPath = os.path.join(os.path.abspath(os.getcwd()), f'{self.resultPath}/model')
        logger.info(f"saving, checkpointPath_toSave={self.checkpointPath_toSave}")
        for ix, agent in enumerate(agents):
            checkpointPath = os.path.join(self.checkpointPath_toSave, f'{ix}')
            if ix == 0:
                train_checkpointer = common.Checkpointer(
                    ckpt_dir=checkpointPath,
                    max_to_keep=checkpoint_max_to_keep,
                    agent=agent,
                    policy=agent.policy,
                    replay_buffer=replay_buffer,
                    global_step=agent.train_step_counter)
            else:
                train_checkpointer = common.Checkpointer(
                    ckpt_dir=checkpointPath,
                    max_to_keep=checkpoint_max_to_keep,
                    agent=agent,
                    policy=agent.policy,
                    global_step=agent.train_step_counter)
            train_checkpointer.save(agent.train_step_counter)

        if replay_buffer.__class__.__name__ in ['ReverbReplayBuffer']:
            reverb_client = reverb.Client(f"localhost:{self.reverb_port}") 
            reverb_checkpointPath = reverb_client.checkpoint()
            logger.info(f"reverb_checkpointPath={reverb_checkpointPath}")


    def run(self, logger, tf_train_env, tf_eval_env, agents, replay_buffer, iterator, checkpoint_max_to_keep=2):

        before_all = time.time()

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        for agent in agents: 
            agent.train = common.function(agent.train)
            agent.train_step_counter.assign(tf.Variable(self.train_step, dtype=tf.int64))  # sync to self.train_step

        # Evaluate the agent's policy once before training.
        avg_return = multiagent_compute_avg_return(tf_eval_env, agents=agents, num_episodes=self.num_episodes_to_eval)
        logger.info(f"before training, avg_return={avg_return}")
        returns = [avg_return]

        # env_step = py_train_env.reset()
        env_step = tf_train_env.reset()
        before = time.time()
        train_step = 0
        for time_step in range(self.num_time_steps):
            self.train_step = train_step  # for epsilon_decay

            # collect a few steps and save in the replay buffer.
            for _ in range(self.num_collect_steps_per_train_step):
                collect_trajectory(logger, tf_train_env, replay_buffer, agents=agents)

            if time_step % self.num_time_steps_to_train == 0:
                # experience, unused_info = next(iterator)
                trajectory, unused_info = iterator.get_next()  # trajectory as tensor
                # logger.debug(f"trajectory={trajectory}")

                trajectories = trajectories_from_merged_trajectory(trajectory, len(agents))
                # logger.debug(f"trajectories={trajectories}")

                train_loss = 0
                for ix, agent in enumerate(agents):
                    loss_info = agent.train(trajectories[ix])
                    train_loss += loss_info.loss

            if time_step % self.num_time_steps_to_log == 0:
                after = time.time()
                logger.info(f'time_step={time_step} loss={train_loss:.3f} time={after-before:.3f}')
                before = after
            if time_step % self.num_time_steps_to_eval == 0:
                avg_return = multiagent_compute_avg_return(tf_eval_env, agents=agents, num_episodes=self.num_episodes_to_eval)
                logger.info(f'time_step={time_step} avg_return={avg_return:.3f}')
                returns.append(avg_return)

            train_step = agent.train_step_counter.numpy()
            if train_step % self.num_train_steps_to_save == 0:
                self.save(logger, agents, replay_buffer, checkpoint_max_to_keep)

        self.save(logger, agents, replay_buffer, checkpoint_max_to_keep)  # once more in last time

        after_all = time.time()
        logger.info(f"total_time={after_all-before_all:.3f}")


