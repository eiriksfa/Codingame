import sys
import math

"""
Wait one round to check patterns, (initialize simulation)
Build Simulation, based on patterns
Find solution:
    Monte-Carlo Search
    Q-Learning
    Optmizitaion (Evolutionary algorithm)
"""

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# width: width of the firewall grid
# height: height of the firewall grid
width, height = [int(i) for i in input().split()]

# game loop
while True:
    # rounds: number of rounds left before the end of the game
    # bombs: number of bombs left
    rounds, bombs = [int(i) for i in input().split()]
    for i in range(height):
        map_row = input()  # one line of the firewall grid

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)

    print("3 0")









import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
from collections import namedtuple

if "../../" not in sys.path:
    sys.path.append("../../")
from lib.envs.mountain_car import MountainCarEnv

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


class NeuralNetwork():
    """
    Neural Network class based on TensorFlow.
    """

    def __init__(self):
        self._build_model()

    def _build_model(self):
        """
        Creates a neural network, e.g. with two
        hidden fully connected layers and 20 neurons each). The output layer
        has #A neurons, where #A is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with
        a learning rate of 0.0005). For initialization, you can simply use a uniform
        distribution (-0.5, 0.5), or something different.
        """
        self.inp = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.inpW = tf.Variable(tf.random_uniform([2, 20], -0.5, 0.5))
        self.inpB = tf.Variable(tf.constant(0.1, shape=[20]))

        self.h1 = tf.nn.relu(tf.matmul(self.inp, self.inpW) + self.inpB)
        self.h1W = tf.Variable(tf.random_uniform([20, 20], -0.5, 0.5))
        self.h1B = tf.Variable(tf.constant(0.1, shape=[20]))

        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.h1W) + self.h1B)
        self.h2W = tf.Variable(tf.random_uniform([20, 3], -0.5, 0.5))
        self.h2B = tf.Variable(tf.constant(0.1, shape=[3]))

        self.out = tf.matmul(self.h2, self.h2W) + self.h2B

        self.y_ = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        #
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)  # GradientDescentOptimizer(learning_rate=0.1)  #
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.out))  # mean?
        self.step = self.trainer.minimize(self.loss)

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        states = np.atleast_2d(states)
        states = np.reshape(states, [len(states), 2])
        feed = {self.inp: states}
        prediction = sess.run(self.out, feed)
        return prediction

    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        pred = self.predict(sess, states)

        # There must be a faster way to do this..
        for i in range(len(actions)):
            pred[i][actions[i]] = targets[i]

        states = np.atleast_2d(states)
        states = np.reshape(states, [len(states), 2])
        sess.run(self.step, feed_dict={self.inp: states, self.y_: pred})

        return self.loss


class TargetNetwork(NeuralNetwork):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, tau=0.001):
        NeuralNetwork.__init__(self)  # scope, summaries_dir
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx, var in enumerate(tf_vars[0:total_vars // 2]):
            op_holder.append(tf_vars[idx + total_vars // 2].assign((var.value
                                                                    () * self.tau) + ((1 - self.tau) * tf_vars[
                idx + total_vars // 2].value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
            sess.run(op)


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    # TODO: What about removing some transitions every now and then?
    def __init__(self):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)

    def next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

    def transition_size(self):
        return len(self._data.states)


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(sess, observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(sess, env, approx, num_episodes, max_time_per_episode, discount_factor=0.99, epsilon=0.1,
               use_experience_replay=False, batch_size=128, target=None):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    Implements the options of online learning or using experience replay and also
    target calculation by target networks, depending on the flags. You can reuse
    your Q-learning implementation of the last exercise.

    Args:
      env: OpenAI environment.
      approx: Action-Value function estimator
      num_episodes: Number of episodes to run for.
      max_time_per_episode: maximum number of time steps before episode is terminated
      discount_factor: gamma, discount factor of future rewards.
      epsilon: Chance to sample a random action. Float betwen 0 and 1.
      use_experience_replay: Indicator if experience replay should be used.
      batch_size: Number of samples per batch.
      target: Slowly updated target network to calculate the targets. Ignored if None.

    Returns:
      An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
    buffer = ReplayBuffer()

    for i_episode in range(num_episodes):
        # The policy we're following
        est = approx if target is None else target
        policy = make_epsilon_greedy_policy(
            est, epsilon, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
        sys.stdout.flush()

        observation = env.reset()
        done = False
        st = time.time()
        while not done and time.time() < st + max_time_per_episode:
            old_observation = observation

            p = policy(sess, observation)
            action = np.random.choice(np.arange(len(p)), p=p)

            observation, reward, done, info = env.step(action)

            # How often, and how long do we update? What about when it randomly picks up one that finishes?
            # (e.g. td = r + discount * finished_reward * np.amax(pred, axis=1) ?)
            if use_experience_replay:
                buffer.add_transition(old_observation, action, observation, reward, done)
                s, a, ns, r, d = buffer.next_batch(batch_size)
                pred = approx.predict(sess, ns)
                td = r + discount_factor * np.amax(pred, axis=1)
                approx.update(sess, s, a, td)
            else:
                # Predict move
                pred = approx.predict(sess, observation)
                # Calc reward
                td = reward + discount_factor * np.max(pred)
                # Update NN
                approx.update(sess, old_observation, action, td)
            stats.episode_rewards[i_episode] += reward

            # Correct use of target network?
            if target is not None:
                target.update(sess)

        stats.episode_lengths[i_episode] = time.time() - st

    return stats


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    fig1.savefig('episode_lengths.png')
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    fig2.savefig('reward.png')
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)


if __name__ == "__main__":
    env = MountainCarEnv()
    approx = NeuralNetwork()
    target = TargetNetwork()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    # # Choose one.
    # stats = q_learning(sess, env, approx, 3000, 10.000)  # env, approx, num_episodes, max_time_per_episode, discount_factor=0.99, epsilon=0.1, use_experience_replay=False, batch_size=128, target=None
    stats = q_learning(sess, env, approx, 40, 10.000, use_experience_replay=True, batch_size=128, target=target)
    plot_episode_stats(stats)
    for _ in range(100):
        state = env.reset()
        for _ in range(1000):
            env.render()
            state, _, done, _ = env.step(np.argmax(approx.predict(sess, [state])))
            if done:
                break
S