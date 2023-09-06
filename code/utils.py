# version 1.1

import numpy as np
import gym

def argmax_with_random_tiebreaking(v: np.array) -> int:
    '''
    Returns the argmax of a 1D numpy array, breaking ties randomly.
    :param v (np.array): A 1D numpy array
    :return: The argmax of v, with ties broken randomly
    '''
    return np.random.choice(np.where(v == v.max())[0])


def sample_integer_from_categorical_distribution(P: np.array) -> int:
    '''
    Sample an integer value from a categorical distribution P.
    :param P: A probability distribution represented as a numpy array with shape [n], where n is the number of possible
              values. That is, for each i in {0, 1, ..., n-1}, P[i] is the probability of sampling i.
    :return: The randomly sampled integer
    '''
    n = P.shape[0]
    return np.random.choice(np.arange(n), p=P)


def episode_with_greedy_policy(env: gym.Env, Q: np.array, discount_factor: float, seed: int=0) -> float:
    '''
    Conducts a single episode in a gym environment, selecting actions with a policy that is greedy with respect to
    learned action-values.
    :param env: The environment with which the agent interacts
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param discount_factor: Discount factor, in (0, 1]
    :param seed: The seed fed to the environment's random number generator
    :return: The return achieved during the episode (i.e. total discounted reward)
    '''

    G = 0.
    terminal = False
    step = 0

    s = env.reset()         # Reset the environment
    env.seed(seed)          # Seed the environment's random number generator
    while not terminal:
        a = argmax_with_random_tiebreaking(Q[s])   # Select the greedy action
        s, r, terminal, _ = env.step(a)     # Apply the action and observe the reward
        G += discount_factor ** step * r    # Add the discounted reward for this step
        step += 1
    return G