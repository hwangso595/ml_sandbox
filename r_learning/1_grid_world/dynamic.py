from grid_world import *
from utils import *
import numpy as np

POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
PRINT_TIME = 3


def init_values(grid):
    values = {}
    for state in grid.all_states():
        values[state] = 0.0
    return values


def init_policy(grid):
    policy = {}
    for state in grid.non_terminal_states():
        policy[state] = np.random.choice(grid.actions[state])
    return policy


def get_expected(probs, values):
    expected_reward = 0
    expected_value = 0
    for (prob, reward, state) in probs:
        expected_reward += prob * reward
        expected_value += prob * values[state]
    return expected_reward, expected_value


def value_iteration(grid, gamma=0.9, epsilon=1e-3, print_grid=False):
    non_terminal_states = grid.non_terminal_states()
    values = init_values(grid)
    policy = init_policy(grid)
    largest_change = epsilon + 10
    iteration = 0
    while largest_change > epsilon:
        largest_change = 0
        for state in non_terminal_states:
            grid.set_state(state)
            best_value = float('-inf')
            best_action = ''
            for action in grid.actions[state]:
                probs = grid.get_transition_probs(action)
                expected_reward, expected_value = get_expected(probs, values)
                action_value = expected_reward + gamma * expected_value
                if action_value > best_value:
                    best_action, best_value = (action, action_value)
            largest_change = max(abs(best_value - values[state]), largest_change)
            values[state] = best_value
            policy[state] = best_action
        if print_grid:
            iteration += 1
            print_data(values, policy, grid, iteration, largest_change)
    return values, policy


def policy_iteration(grid, gamma=1.0, epsilon=1e-3, print_grid=False):
    non_terminal_states = grid.non_terminal_states()
    values = init_values(grid)
    policy = init_policy(grid)
    policy_stable = False
    # policy evaluation
    iteration = 0
    while not policy_stable:
        largest_change = epsilon + 1
        while largest_change > epsilon:
            largest_change = 0
            for state in non_terminal_states:
                grid.set_state(state)
                value = values[state]
                probs = grid.get_transition_probs(policy[state])
                expected_reward, expected_value = get_expected(probs, values)
                values[state] = expected_reward + gamma * expected_value
                largest_change = max(abs(value-values[state]), largest_change)
        # policy improvement
        policy_stable = True
        for state in non_terminal_states:
            old_action = policy[state]
            grid.set_state(state)
            best_value = float('-inf')
            best_action = policy[state]
            for action in grid.actions[state]:
                probs = grid.get_transition_probs(action)
                expected_reward, expected_value = get_expected(probs, values)
                state_value = expected_reward + gamma * expected_value
                if state_value > best_value:
                    best_value = state_value
                    best_action = action
            policy[state] = best_action
            if best_action != old_action:
                policy_stable = False
        if print_grid:
            iteration += 1
            print_data(values, policy, grid, iteration, largest_change)

    return values, policy


if __name__ == '__main__':
    g = standard_grid(obey_prob=1, step_cost=-2.0001)
    # v, p = value_iteration(g, gamma=0.9, epsilon=1e-3, print_grid=True)
    v, p = policy_iteration(g, gamma=0.9, epsilon=1e-3, print_grid=True)
    print_values(v, g)
    print()
    print_policy(p, g)
