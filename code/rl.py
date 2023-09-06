# version 1.1

import gym
import numpy as np

from utils import *
from wumpus_env import *


def select_action(strategy: str, s: int, Q: np.array, N_sa: np.array, epsilon: float=0.05, T: float=10., N_e: int=5,
                  R_plus: float=999.):
    '''
    Selects an action for the current state according to the desired strategy. This may be of use to you for the
    Q-learning and SARSA functions.
    :param strategy: Action selection strategy for exploration - one of {"optimistic", "softmax", "epsilon_greedy"}
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param N_sa: A [n_states, n_actions] array indicating the number of times that each state/action pair has been visited
    :param epsilon: The probability for selecting a random action (in [0, 1])
    :param T: The temperature for softmax action selection
    :param N_e: Number of times a state-action pair is visited before expected utility is used instead of optimistic estimates
    :param R_plus: The best possible reward obtainable in any state
    '''

    if strategy == "optimistic":
        return select_action_optimistically(s, Q, N_sa, N_e, R_plus)
    elif strategy == "softmax":
        return select_action_softmax(s, Q, T)
    else:
        return select_action_epsilon_greedy(s, Q, epsilon)


def select_action_epsilon_greedy(s: int, Q: np.array, epsilon: float=0.1) -> int:
    '''
    With probability epsilon, select a random action. Otherwise, select the greedy action.
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param epsilon: The probability for selecting a random action (in [0, 1])
    '''

    samp_int = sample_integer_from_categorical_distribution(np.array([epsilon, 1 - epsilon]))
    if samp_int == 0:
        n_actions = len(Q[s])
        prob_rand = 1/n_actions
        prob_actions = np.full(n_actions, prob_rand)
        a = sample_integer_from_categorical_distribution(prob_actions)
    else:
        a = argmax_with_random_tiebreaking(Q[s])
    return a


def select_action_softmax(s: int, Q: np.array, T: float=10.0) -> int:
    '''
    Select an action via softmax selection using the Gibbs/Boltzmann distribution. Assumes that the temperature is
    always nonzero.
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param T: The temperature (T != 0)
    '''

    c = np.max(Q[s])
    prob_numerator = np.exp((Q[s] - c)/T)
    prob_denom = np.sum(prob_numerator)
    prob_actions = prob_numerator / prob_denom
    a = sample_integer_from_categorical_distribution(prob_actions)
    return a


def select_action_optimistically(s: int, Q: np.array, N_sa: np.array, N_e: int=5, R_plus: np.float=999.0) -> int:
    '''
    Use optimistic utility estimates to select an action. If the
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param N_sa: A [n_states, n_actions] array indicating the number of times that each state/action pair has been visited
    :param N_e: Number of times a state-action pair is visited before expected utility is used instead of optimistic estimates
    :param R_plus: The best possible reward obtainable in any state
    '''

    a = argmax_with_random_tiebreaking(np.where(N_sa[s] < N_e, R_plus, Q[s]))
    return a


def active_q_learning(env: WumpusWorld, Q_init: np.array, n_episodes: int, action_selection: str='optimistic',
                      discount_factor=0.99, alpha: float=0.5, epsilon: float=0.1, T: float=1., N_e: int=3,
                      R_plus: float=999.) -> (np.array, np.array):
    '''
    Conducts active Q-learning to learn optimal Q-values. Q-values are updated during each step for a fixed number of
    episodes.
    :param env: The environment with which the agent interacts
    :param Q_init [env.n_states, env.n_actions]: Initial action values
    :param n_episodes: The number of training episodes during which experience can be collected to learn the Q-values
    :param action_selection: Action selection strategy for exploration - one of {"optimistic", "softmax", "epsilon_greedy"}
    :param discount_factor: Discount factor, in (0, 1]
    :param alpha: Learning rate. alpha > 0.
    :param epsilon: The probability for selecting a random action (in [0, 1])
    :param T: The temperature for softmax action selection
    :param N_e: Number of times a state-action pair is visited before expected utility is used instead of optimistic estimates
    :param R_plus: The best possible reward obtainable in any state
    :return: (Final Q-values after convergence [env.n_states, env.n_actions],
              Total undiscounted reward obtained in each episode [n_episodes])
    '''

    Q = Q_init
    N_sa = np.zeros((env.n_states, env.n_actions))
    episode_rewards = np.zeros((n_episodes))

    for i in range(n_episodes): # for each of E episodes
        curr_s = env.reset() # get initial state
        terminal = False
        while not terminal: # keep going through states until the current state is terminal
            curr_a = select_action(action_selection, curr_s, Q, N_sa, epsilon, T, N_e, R_plus)
            step_return = env.step(curr_a) # returns tuple
            next_s = step_return[0]
            r = step_return[1]
            terminal = step_return[2]
            episode_rewards[i] += r # update rewards
            Q[curr_s][curr_a] += alpha * (r + discount_factor * np.max(Q[next_s]) - Q[curr_s][curr_a]) # update Q(s, a)
            N_sa[curr_s][curr_a] += 1
            curr_s = next_s

    return Q, episode_rewards


def active_sarsa(env: WumpusWorld, Q_init: np.array, n_episodes: int, action_selection: str='optimistic',
                 discount_factor: float=0.99, alpha: float=0.5, epsilon: float=0.1, T: float=1., N_e: int=3,
                 R_plus: float=999.) -> (np.array, np.array):
    '''
    Conducts active SARSA to learn optimal Q-values. Q-values are updated during each step for a fixed number of
    episodes.
    :param env: The environment with which the agent interacts
    :param Q_init: Initial action values
    :param n_episodes: The number of training episodes during which experience can be collected to learn the Q-values
    :param action_selection: Action selection strategy for exploration - one of {"optimistic", "softmax", "epsilon_greedy"}
    :param discount_factor: Discount factor, in (0, 1]
    :param alpha: Learning rate. alpha > 0.
    :param epsilon: The probability for selecting a random action (in [0, 1])
    :param T: The temperature for softmax action selection
    :param N_e: Number of times a state-action pair is visited before expected utility is used instead of optimistic estimates
    :param R_plus: The best possible reward obtainable in any state
    :return: (Final Q-values after convergence [env.n_states, env.n_actions],
              Total undiscounted reward obtained in each episode [n_episodes])
    '''

    Q = Q_init
    N_sa = np.zeros((env.n_states, env.n_actions))
    episode_rewards = np.zeros((n_episodes))

    for i in range(n_episodes): # for each of E episodes
        curr_s = env.reset() # get initial state
        curr_a = select_action(action_selection, curr_s, Q, N_sa, epsilon, T, N_e, R_plus)
        terminal = False
        while not terminal: # keep going through states until the current state is terminal
            step_return = env.step(curr_a) # returns tuple
            next_s = step_return[0]
            r = step_return[1]
            terminal = step_return[2]
            episode_rewards[i] += r # update rewards
            next_a = select_action(action_selection, next_s, Q, N_sa, epsilon, T, N_e, R_plus)
            Q[curr_s][curr_a] += alpha * (r + discount_factor * Q[next_s][next_a] - Q[curr_s][curr_a]) # update Q(s, a)
            N_sa[curr_s][curr_a] += 1
            curr_s = next_s
            curr_a = next_a
    
    return Q, episode_rewards