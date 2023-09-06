from rl import *
import matplotlib.pyplot as plt

T = 0.1
N_e = 2
R_plus = 999
n_episodes = 50000
strategy = 'softmax'
N = 50
rewards_all = [0,0,0]

env = WumpusWorld()
for i in range(3): # 3 runs
    env.seed(i)
    Q_init = np.zeros((env.n_states, env.n_actions))
    temp_Q, rewards = active_sarsa(env=env, Q_init=Q_init, n_episodes=n_episodes, action_selection=strategy, T=T, N_e=N_e, R_plus=R_plus)
    rewards_all[i] = rewards
rewards_avg = np.array([(rewards1 + rewards2 + rewards3)/3 for rewards1, rewards2, rewards3 in zip(rewards_all[0], rewards_all[1], rewards_all[2])])
rewards_mov_avg = np.convolve(rewards_avg, np.ones(N)/N, mode = 'valid')
plt.plot(list(range(50, 50001, 1)), rewards_mov_avg, color="red")
plt.title("SARSA with optimistic utility exploration")
plt.xlabel("Episode number")
plt.ylabel("Total undiscounted reward averaged across 3 runs")
plt.xlim(50 , 50000)
plt.show()