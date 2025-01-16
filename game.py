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

# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics
def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        env_step = environment.reset()
        episode_return = 0.0

        while not env_step.is_last():
            action_step = policy.action(env_step)
            # print(f"in compute_avg_return, action_step={action_step}", flush=True)
            # print(f"in compute_avg_return, action={action_step.action}", flush=True)
            env_step = environment.step(action_step.action)
            episode_return += env_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


class Game:
    def __init__(self, config):
        self.reverb_port = config['reverb_port']
        self.num_time_steps = config['num_time_steps'] if config['num_time_steps'] > 0 else sys.maxsize
        self.num_time_steps_to_log = config['num_time_steps_to_log']  # max((int) (self.num_time_steps / config['num_logs']), 1)
        self.num_time_steps_to_eval = config['num_time_steps_to_eval']  # max((int) (self.num_time_steps / config['num_evals']), 1)
        self.num_time_steps_to_train = config['num_time_steps_to_train']
        self.num_train_steps_to_save = config['num_train_steps_to_save']
        self.num_episodes_to_eval = config['num_episodes_to_eval']


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
        agent.train_step_counter.assign(tf.Variable(0, dtype=tf.int64))

        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(tf_eval_env, agent.policy, self.num_episodes_to_eval)
        logger.info(f"before training, avg_return={avg_return:.3f}")
        returns = [avg_return]

        env_step = py_train_env.reset()
        before = time.time()
        for time_step in range(self.num_time_steps):

            # drive env multiple steps and save env_steps to replay_buffer.
            if driver.__class__.__name__ in ['PyDriver']:
                env_step, _ = driver.run(env_step)
            elif driver.__class__.__name__ in ['DynamicStepDriver','DynamicEpisodeDriver']:
                driver.run()

            if time_step % self.num_time_steps_to_train == 0:
                # experience, unused_info = next(iterator)
                experience, unused_info = iterator.get_next()  # experience as tensor
                logger.debug(f"experience={experience}")
                loss_info = agent.train(experience)
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

