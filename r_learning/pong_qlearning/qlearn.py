import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from wrappers import make_env
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FILE_WEIGHTS = "./weights.h5"
FILE_OPTIMIZER_WEIGHTS = "./optimizer.pkl"
MAX_BUFFER_SIZE = 200
MAX_STEPS = 50000


def create_model(input_shape, num_actions):
    model = models.Sequential([
        layers.Conv2D(32, 8, 4, activation='relu', input_shape=input_shape),
        layers.Conv2D(64, 4, 2, activation='relu'),
        layers.Conv2D(64, 3, 1, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_actions)
    ])
    return model


def move(q, epsilon, state, actions):
    p = np.random.uniform()
    if p < epsilon:
        return actions.sample()
    else:
        q_values = q(tf.convert_to_tensor([state]), training=False)[0]
        return np.argmax(q_values)


def save_model(model, optimizer):
    model.save_weights(FILE_WEIGHTS)
    symbolic_weights = getattr(optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open(FILE_OPTIMIZER_WEIGHTS, 'wb') as f:
        pickle.dump(weight_values, f)


def load_models(env, alpha):
    model = create_model(env.observation_space.shape, env.action_space.n)
    model_target = create_model(env.observation_space.shape, env.action_space.n)
    optimizer = optimizers.Adam(learning_rate=alpha, clipnorm=1.0)
    if os.path.exists(FILE_WEIGHTS) and os.path.exists(FILE_OPTIMIZER_WEIGHTS):
        model.load_weights(FILE_WEIGHTS)
        model_target.load_weights(FILE_WEIGHTS)
        with open(FILE_OPTIMIZER_WEIGHTS, 'rb') as f:
            weight_values = pickle.load(f)
        optimizer.set_weights(weight_values)
    return model, model_target, optimizer


def q_learning(epsilon_start=1, epsilon_decay=0.0002, update_every_n=3000, batch_size=32, alpha=0.0002, gamma=0.99):
    env = make_env(DEFAULT_ENV_NAME)
    loss_func = losses.MeanAbsoluteError()
    q, q_target, optimizer = load_models(env, alpha)
    state = env.reset()
    buffer = []
    net_point = 0
    num_game = 1
    step = 0
    while True:
        epsilon = epsilon_start/(1 + epsilon_decay*step)
        action = move(q, epsilon, state, env.action_space)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        step += 1
        buffer.append([state, action, reward, done, next_state])
        if len(buffer) > MAX_BUFFER_SIZE:
            del buffer[:1]
        net_point += reward
        state = next_state
        if done:
            state = env.reset()
            print("game: %i, net point: %f" % (num_game, net_point))
            net_point = 0
            num_game += 1

        indices = np.random.choice(range(len(buffer)), size=batch_size)
        state_batch = np.array([buffer[i][0] for i in indices])
        action_batch = np.array([buffer[i][1] for i in indices])
        reward_batch = np.array([buffer[i][2] for i in indices])
        done_batch = tf.convert_to_tensor([float(buffer[i][3]) for i in indices])
        next_state_batch = np.array([buffer[i][4] for i in indices])

        # get y target and y prediction to apply gradients
        y_true = reward_batch + gamma * tf.reduce_max(q_target.predict(next_state_batch), 1) * (1 - done_batch)
        action_one_hot = tf.one_hot(action_batch, env.action_space.n)
        with tf.GradientTape() as tape:
            y_pred = tf.reduce_sum(tf.multiply(q(state_batch), action_one_hot), 1)
            loss = loss_func(y_true, y_pred)
        grads = tape.gradient(loss, q.trainable_variables)
        optimizer.apply_gradients(zip(grads, q.trainable_variables))
        for count, reward in enumerate(reward_batch):
            if reward:
                print(reward_batch[count], q.predict(tf.convert_to_tensor([state_batch[count]])))
                break

        if step % update_every_n == 0 and step:
            q_target.set_weights(q.get_weights())
            save_model(q, optimizer)


if __name__ == '__main__':
    q_learning()
