import os
import numpy as np
import gym
from gym import wrappers
import time
import matplotlib.pyplot as plt

# import pybullet_envs

ENV_NAME = 'BipedalWalker-v3'
# ENV_NAME = 'HalfCheetahBulletEnv-v0'

MONITOR_DIR = os.path.join('videos', ENV_NAME)
MAX_STEPS = 2000
MAX_GEN = 200
RECORD_EVERY_N_GEN = 3200

NORMAL_DATA = './standardization.npy'
WEIGHTS_DATA = './weights.npy'

env = gym.make(ENV_NAME)
should_record = lambda x: x % RECORD_EVERY_N_GEN == 0
# env = wrappers.Monitor(env, directory=MONITOR_DIR, video_callable=should_record, force=True)
action_size = env.action_space.shape[0]
input_size = env.observation_space.shape[0]


def generate_deltas(num_deltas):
    return np.random.randn(action_size, input_size, num_deltas)


class Normalizer():
    # Normalizes the inputs
    def __init__(self, nb_inputs, n=None, mean=None, mean_diff=None):

        if n is not None:
            assert mean is not None and mean_diff is not None
            self.n = n
            self.mean = mean
            self.mean_diff = mean_diff
        else:
            self.n = np.zeros((nb_inputs, 1))
            self.mean = np.zeros((nb_inputs, 1))
            self.mean_diff = np.zeros((nb_inputs, 1))
        self.var = np.zeros((nb_inputs, 1))

    def observe(self, x):
        assert x.shape == self.n.shape
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / (self.n - 1).clip(min=1)).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


def play_episode(theta, normalizer, render=False):
    total_reward = 0
    state = env.reset()
    for num_steps in range(MAX_STEPS):
        state = np.reshape(state, (len(state), 1))
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = theta.dot(state)
        state, reward, done, _ = env.step(action)
        if render:
            env.render()
        total_reward += np.clip(reward, a_min=-1, a_max=1)
        if done:
            break
    return total_reward


def update_theta(theta, deltas, rollouts, r_std, learning_rate):
    step = 0
    for r_pos, r_neg, episode_num in rollouts:
        step += (r_pos - r_neg) * deltas[:, :, episode_num]
    return theta + learning_rate * step / (len(rollouts) * r_std)


def load_data():
    if os.path.exists(WEIGHTS_DATA) and os.path.exists(NORMAL_DATA):
        print('data loaded')
        with open(WEIGHTS_DATA, 'rb') as f:
            theta = np.load(f)
        with open(NORMAL_DATA, 'rb') as f:
            n = np.load(f)
            mean = np.load(f)
            mean_diff = np.load(f)
        normalizer = Normalizer(input_size, n, mean, mean_diff)
    else:
        theta = np.random.randn(action_size, input_size) * np.sqrt(2 / input_size)
        normalizer = Normalizer(input_size)
    return theta, normalizer


def save_data(theta, normalizer):
    with open(WEIGHTS_DATA, 'wb') as f:
        np.save(f, theta)
    with open(NORMAL_DATA, 'wb') as f:
        np.save(f, normalizer.n)
        np.save(f, normalizer.mean)
        np.save(f, normalizer.mean_diff)


def ars(num_deltas=16, num_best_deltas=15, exploration_noise=0.03, learning_rate=0.02):
    # HE INITIALIZATION
    reward_per_gen = []
    thetas, normalizer = load_data()
    for generation in range(MAX_GEN):
        deltas = generate_deltas(num_deltas)
        rollouts = []
        r = np.zeros((2, num_deltas))
        current_time = time.time()
        for episode in range(num_deltas):
            theta_pos = thetas + exploration_noise*deltas[:, :, episode]
            theta_neg = thetas - exploration_noise*deltas[:, :, episode]
            r[0, episode] = play_episode(theta_pos, normalizer)
            r[1, episode] = play_episode(theta_neg, normalizer)
            rollouts.append((r[0, episode], r[1, episode], episode))
        print(time.time()-current_time)
        r_std = np.std(r, ddof=1)
        rollouts_new = sorted(rollouts, key=lambda x: max(x[0], x[1]), reverse=True)[:num_best_deltas]
        thetas = update_theta(thetas, deltas, rollouts_new, r_std, learning_rate)

        reward_per_gen.append(play_episode(thetas, normalizer))
        print('Generation: ' + str(generation + 1) + '| total reward: ' + str(reward_per_gen[generation]))

    save_data(thetas, normalizer)

    plt.plot(reward_per_gen)
    plt.show()


def play_latest():
    theta, normalizer = load_data()
    print(play_episode(theta, normalizer, True))


if __name__ == '__main__':
    # ars()
    play_latest()

