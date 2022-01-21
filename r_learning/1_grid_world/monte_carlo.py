from grid_world import *
from utils import *
import numpy as np

POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def init_quality(grid):
    # initialize q[state, action] values with all zeros
    quality = {}
    for state in grid.non_terminal_states():
        for action in POSSIBLE_ACTIONS:
            quality[state, action] = 0.0
    return quality


def init_values(grid):
    # initialize values[state] with all zeros
    values = {}
    for state in grid.all_states():
        values[state] = 0.0
    return values


def init_policy(grid):
    # initialize policy[state] with random actions
    policy = {}
    for state in grid.non_terminal_states():
        policy[state] = np.random.choice(POSSIBLE_ACTIONS)
    return policy


def init_returns(grid):
    # initialize returns[state, action] with empty arrays
    returns = {}
    for state in grid.non_terminal_states():
        for action in POSSIBLE_ACTIONS:
            returns[state, action] = []
    return returns


def play_game(policy, grid, epsilon, max_move=100):
    # play a game until game is over or number of moves pass max_move.
    # Returns episode of (state, action, reward)
    episode = []
    move_num = 0
    while not grid.game_over() and move_num < max_move:
        current_state = grid.current_state()
        # move according to epsilon greedy policy
        p = np.random.uniform()
        action = ''
        if p < 1 - epsilon:
            action = policy[current_state]
        else:
            action = np.random.choice(POSSIBLE_ACTIONS)
        reward = grid.move(action)
        episode.append((current_state, action, reward))
        move_num += 1
    return episode


def epsilon_greedy(grid, gamma=0.9, iterations=1000, epsilon=0.2, print_grid=False):
    policy = init_policy(grid)
    values = init_values(grid)
    q = init_quality(grid)
    returns = init_returns(grid)
    start_state = grid.current_state()
    plays = 0
    for i in range(iterations):
        g = 0
        grid.set_state(start_state)
        episode = play_game(policy, grid, epsilon)
        plays += 1
        non_duplicate_states = set()
        for state, move, reward in reversed(episode):
            if (state, move) not in non_duplicate_states:
                # If we haven't seen this state,action pair in the game, append G to returns[s, a]
                non_duplicate_states.add((state, move))
                g = reward + gamma * g
                returns[state, move].append(g)
                # set q[state, move] = mean(returns[state, move])
                q[state, move] = (g + (len(returns[state, move])-1) * q[state, move])/len(returns[state, move])
                # set new policy[state] = argmax_[a](q[state, action])
                # set new value[state] = max_[a](q[state, action])
                if state in grid.non_terminal_states():
                    best_action, best_value = '', float('-inf')
                    for action in POSSIBLE_ACTIONS:
                        if q[state, action] > best_value:
                            best_action = action
                            best_value = q[state, action]
                    policy[state] = best_action
                    values[state] = best_value
        if print_grid and plays % 100 == 0:
            print_data(values, policy, grid, plays)
    return values, policy


if __name__ == '__main__':
    grid = standard_grid()
    v, p = epsilon_greedy(grid, print_grid=True)
    print_values(v, grid)
    print_policy(p, grid)
