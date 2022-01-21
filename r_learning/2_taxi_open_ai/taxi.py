import random
import numpy as np
import matplotlib.pyplot as plt
import gym
from utils import checkpoint, play

GAME = 'Taxi-v3'
env = gym.make(GAME)

CHECKPOINT_DIR = 'checkpoints'
MAX_STEPS = 200 # maximum steps in an episode, 200 for Taxi-v2

NUM_EPISODES = 50000
GAMMA = 0.95  # Discount factor from Bellman Equation
START_ALPHA = 0.1  # Learning rate, how much we update our Q table each step
ALPHA_TAPER = 0.01 # How much our adaptive learning rate is adjusted each update
START_EPSILON = 1.0  # Probability of random action
EPSILON_TAPER = 0.01 # How much epsilon is adjusted each step

def epsilon_greedy(action, epsilon):
    p = np.random.uniform()
    if p < 1 - epsilon:
        return action
    else:
        return env.action_space.sample()


def q_learning(num_episodes=NUM_EPISODES, gamma=GAMMA, start_alpha=START_ALPHA, alpha_taper=ALPHA_TAPER,
               start_epsilon=START_EPSILON, epsilon_taper=EPSILON_TAPER, print_data=True):
    obs_dim = env.observation_space.n  # size of our state
    action_dim = env.action_space.n  # number of actions
    # Initialize our Q table
    Q = np.zeros((obs_dim, action_dim))
    count = np.zeros((obs_dim, action_dim))
    policy = np.random.choice(action_dim, (obs_dim, 1))
    for i in range(num_episodes + 1):
        state = env.reset()
        if i % 10000 == 0:
            cp_file = checkpoint(Q, CHECKPOINT_DIR, GAME, i)
            if print:
                print("Saved checkpoint to: ", cp_file)
        for step_num in range(MAX_STEPS):
            epsilon = start_epsilon/(1 + epsilon_taper * step_num)
            action = epsilon_greedy(policy[state][0], epsilon)
            count[state][action] += 1
            alpha = start_alpha/(1 + count[state, action] * alpha_taper)

            next_state, reward, done, _ = env.step(action)
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
            if done:
                break
    return cp_file

if __name__ == '__main__':
    file_path = q_learning()


# to play, python replay.py 'file path'
#ex. python replay.py checkpoints\Taxi-v3_50000.npy