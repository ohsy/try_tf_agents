"""
game with tf-agents
2025.1
Sangyeop Oh
"""

import sys
import time
# import reverb
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.trajectories import from_transition


def collect_trajectory(logger, environment, replay_buffer, policy):
    env_step = environment.current_time_step()
    action_step = policy.action(env_step)
    next_env_step = environment.step(action_step.action)
    trajectory = from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(trajectory)


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        env_step = environment.reset()
        episode_return = 0.0

        while not env_step.is_last():
            action_step = policy.action(env_step)
            env_step = environment.step(action_step.action)
            episode_return += env_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


class Game:
    def __init__(self, config):
        self.num_env_steps_to_collect_per_time_step = config['num_env_steps_to_collect_per_time_step']
        self.reverb_port = config['reverb_port']
        self.num_time_steps = config['num_time_steps'] if config['num_time_steps'] > 0 else sys.maxsize
        self.num_time_steps_to_log = config['num_time_steps_to_log']  # max((int) (self.num_time_steps / config['num_logs']), 1)
        self.num_time_steps_to_eval = config['num_time_steps_to_eval']  # max((int) (self.num_time_steps / config['num_evals']), 1)
        self.num_time_steps_to_train = config['num_time_steps_to_train']
        self.num_train_steps_to_save = config['num_train_steps_to_save']
        self.num_episodes_to_eval = config['num_episodes_to_eval']
        self.train_step = 0  # for epsilon decay

        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.num_batches = self.replay_buffer.num_frames() // self.batch_size


    def save(self, logger, agent, checkpointer, reverb_client):
        checkpointer.save(agent.train_step_counter)
        if reverb_client is not None:
            reverb_checkpointPath_to_restore_later = reverb_client.checkpoint()
            logger.info(f"reverb_checkpointPath_to_restore_later={reverb_checkpointPath_to_restore_later}")
        logger.info(f"agent and replay_buffer saved")


    def run(self, logger, py_train_env, tf_eval_env, agent, replay_buffer, iterator, driver, checkpointer, reverb_client):

        before_all = time.time()
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        agent.train = common.function(agent.train)

        # Reset the train step.
        agent.train_step_counter.assign(tf.Variable(self.train_step, dtype=tf.int64))  # sync to self.train_step

        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(tf_eval_env, agent.policy, self.num_episodes_to_eval)
        logger.info(f"before training, avg_return={avg_return:.3f}")
        returns = [avg_return]

        env_step = py_train_env.reset()
        before = time.time()
        train_step = 0
        for time_step in range(self.num_time_steps):
            self.train_step = train_step  # for epsilon_decay

            # collect trajectories from env and fill in replay buffer
            if driver is None:
                for _ in range(self.num_env_steps_to_collect_per_time_step):
                    collect_trajectory(logger, tf_train_env, replay_buffer, agent.collect_policy)
            elif driver.__class__.__name__ in ['PyDriver']:
                env_step, _ = driver.run(env_step)
            elif driver.__class__.__name__ in ['DynamicStepDriver','DynamicEpisodeDriver']:
                driver.run()

            if time_step % self.num_time_steps_to_train == 0:
                # trajectory, unused_info = next(iterator)
                trajectory, unused_info = iterator.get_next()  # trajectory as tensor
                logger.debug(f"trajectory={trajectory}")
                loss_info = agent.train(trajectory)
                train_loss = loss_info.loss
                if time_step % self.num_time_steps_to_log == 0:
                    after = time.time()
                    logger.info(f'time_step={time_step} loss={train_loss:.3f} time={after-before:.3f}')
                    before = after

            if time_step % self.num_time_steps_to_eval == 0:
                avg_return = compute_avg_return(tf_eval_env, agent.policy, self.num_episodes_to_eval)
                logger.info(f'time_step={time_step} avg_return={avg_return:.3f}')
                returns.append(avg_return)

            train_step = agent.train_step_counter.numpy()
            if train_step % self.num_train_steps_to_save == 0:
                self.save(logger, agent, checkpointer, reverb_client)

        self.save(logger, agent, checkpointer, reverb_client)  # once more in last time

        after_all = time.time()
        logger.info(f"total_time={after_all-before_all:.3f}")

    def train_and_save_agent(self, logger, agent, replay_buffer, iterator, checkpointer, reverb_client):
        before = time.time()
        for epoch in range(self.num_epochs):
            logger.info(f'epoch={epoch}')
            for time_step in range(self.num_batches):
                trajectory, unused_info = iterator.get_next()  # trajectory as tensor
                logger.debug(f"trajectory={trajectory}")
                loss_info = agent.train(trajectory)
                train_loss = loss_info.loss
                if time_step % self.num_time_steps_to_log == 0:
                    after = time.time()
                    logger.info(f'time_step={time_step} loss={train_loss:.3f} time={after-before:.3f}')
                    before = after
        self.save(logger, agent, checkpointer, reverb_client)

