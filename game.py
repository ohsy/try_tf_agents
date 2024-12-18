"""
game with tf-agents
"""

import time
# import reverb
import tensorflow as tf
from tf_agents.utils import common

# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics
def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            # print(f"in compute_avg_return, action_step={action_step}", flush=True)
            # print(f"in compute_avg_return, action={action_step.action}", flush=True)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


class Game:
    def __init__(self, config, checkpointPath_toSave):
        self.checkpointPath_toSave = checkpointPath_toSave
        self.reverb_port = config['reverb_port']
        self.num_train_steps = config['num_train_steps']
        self.num_train_steps_to_log = max((int) (self.num_train_steps / 250), 1)
        self.num_train_steps_to_eval = max((int) (self.num_train_steps / 50), 1)
        self.num_train_steps_to_save_model = max((int) (self.num_train_steps / 5), 1)
        self.num_episodes_to_eval = config['num_episodes_to_eval']

    def run(self, logger, py_train_env, tf_eval_env, agent, replay_buffer, iterator, driver):

        before_all = time.time()
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        agent.train = common.function(agent.train)

        # Reset the train step.
        agent.train_step_counter.assign(tf.Variable(0, dtype=tf.int64))

        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(tf_eval_env, agent.policy, self.num_episodes_to_eval)
        logger.info(f"before training, avg_return={avg_return}")
        returns = [avg_return]

        time_step = py_train_env.reset()
        before = time.time()
        for _ in range(self.num_train_steps):

            # Collect a few steps and save to the replay buffer.
            if driver.__class__.__name__ in ['PyDriver']:
                time_step, _ = driver.run(time_step)
            elif driver.__class__.__name__ in ['DynamicStepDriver','DynamicEpisodeDriver']:
                driver.run()

            # experience, unused_info = next(iterator)
            experience, unused_info = iterator.get_next()  # experience as tensor
            logger.debug(f"experience={experience}")
            loss_info = agent.train(experience)
            train_loss = loss_info.loss

            train_step = agent.train_step_counter.numpy()
            if train_step % self.num_train_steps_to_log == 0:
                after = time.time()
                logger.info(f'train_step={train_step} loss={train_loss:.3f} time={after-before:.3f}')
                before = after
            if train_step % self.num_train_steps_to_eval == 0:
                avg_return = compute_avg_return(tf_eval_env, agent.policy, self.num_episodes_to_eval)
                logger.info(f'train_step={train_step} avg_return={avg_return:.3f}')
                returns.append(avg_return)

        after_all = time.time()
        logger.info(f"total_time={after_all-before_all:.3f}")


        # checkpointPath = os.path.join(os.path.abspath(os.getcwd()), f'{self.resultPath}/model')
        logger.info(f"saving, checkpointPath_toSave={self.checkpointPath_toSave}")
        train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpointPath_toSave,
            max_to_keep=2,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter)
        train_checkpointer.save(agent.train_step_counter)

        if replay_buffer.__class__.__name__ in ['ReverbReplayBuffer']:
            reverb_client = reverb.Client(f"localhost:{self.reverb_port}") 
            reverb_checkpointPath = reverb_client.checkpoint()
            logger.info(f"reverb_checkpointPath={reverb_checkpointPath}")

