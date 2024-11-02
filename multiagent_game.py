"""
game with tf-agents
"""

import time
import reverb
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.trajectories import Trajectory


def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)


def merge_action(actions):
    """
    actions: list of actions
    """
    return actions


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics
def multiagent_compute_avg_return(environment, agents, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            actions = []
            for agent in agents:
                action_step = policy.action(time_step)
                actions.append(action_step.action)
            merged_action = merge_action(actions)
            time_step = environment.step(merged_action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


class MultiAgentGame:
    def __init__(self, config, checkpointPath_toSave):
        self.checkpointPath_toSave = checkpointPath_toSave
        self.reverb_port = config['reverb_port']
        self.num_train_steps = config['num_train_steps']
        self.num_train_steps_to_log = config['num_train_steps_to_log']
        self.num_train_steps_to_eval = config['num_train_steps_to_eval']
        self.num_train_steps_to_save_model = config['num_train_steps_to_save_model']
        self.num_episodes_to_eval = config['num_episodes_to_eval']

    def run(self, logger, py_train_env, tf_eval_env, agents, replay_buffer, iterator, driver):

        before_all = time.time()
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        for agent in agents: 
            agent.train = common.function(agent.train)
            agent.train_step_counter.assign(tf.Variable(0, dtype=tf.int64))

        # Evaluate the agent's policy once before training.
        avg_return = multiagent_compute_avg_return(tf_eval_env, agents, self.num_episodes_to_eval)
        logger.info(f"before training, avg_return={avg_return}")
        returns = [avg_return]

        time_step = py_train_env.reset()
        before = time.time()
        for _ in range(self.num_train_steps):

            # Collect a few steps and save to the replay buffer.
            for _ in range(num_collect_steps_per_train_step):
                # collect_step(tf_train_env, agent.collect_policy)
                time_step = tf_train_env.current_time_step()
                # action_step = policy.action(time_step)
                actions = []
                for agent in agents:
                    action_step = agent.collect_policy.action(time_step)
                    actions.append(action_step.action)
                merged_action = merge_action(actions)
                # next_time_step = environment.step(action_step.action)
                next_time_step = environment.step(merged_action)
                traj = trajectory.from_transition(time_step, action_step, next_time_step)

                # Add trajectory to the replay buffer
                replay_buffer.add_batch(traj)

            # experience, unused_info = next(iterator)
            experience, unused_info = iterator.get_next()  # experience as tensor
            logger.debug(f"experience={experience}")
            trajectories = []
            train_loss = 0
            for ix, agent in enumerate(agents):
                trajectories.append(
                    Trajectory(
                        experience.step_type,
                        experience.observation,
                        experience.action[:, :, ix],
                        experience.policy_info,
                        experience.next_step_type,
                        experience.discount))
                loss_info = agent.train(trajectories[ix])
                train_loss += loss_info.loss
            logger.debug(f"trajectories={trajectories}")

            train_step = agent.train_step_counter.numpy()
            if train_step % self.num_train_steps_to_log == 0:
                after = time.time()
                logger.info(f'train_step={train_step}: loss={train_loss:.3f}, time={after-before:.3f}')
                before = after
            if train_step % self.num_train_steps_to_eval == 0:
                avg_return = multiagent_compute_avg_return(tf_eval_env, agents, self.num_episodes_to_eval)
                logger.info(f'train_step={train_step}: average return={avg_return:.3f}')
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

