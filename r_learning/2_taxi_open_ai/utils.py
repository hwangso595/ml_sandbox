import numpy as np
import gym
import os

GAME = 'Taxi-v3'
env = gym.make(GAME)
MAX_STEPS = 300


def play(cp_file):
    Q = np.load(cp_file)
    total_reward = 0
    state = env.reset()
    env.render()

    for step in range(MAX_STEPS):
        prevState = state
        action = np.argmax(Q[state])
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        if done:
            break
    print('Total reward:', total_reward)


def mkdir(name):
    base = os.getcwd()
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def checkpoint(data, dir, filename, step):
    path = mkdir(dir)
    file_path = os.path.join(path, filename + '_' + str(step) + '.npy')
    np.save(file_path, data)
    return file_path
