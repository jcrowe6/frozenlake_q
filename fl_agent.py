import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes()

#env = gym.make("FrozenLake-v1", render_mode="human")
env = gym.make("FrozenLake-v1")
n_observations = env.observation_space.n
n_actions = env.action_space.n

#Initialize the Q-table to 0
Q_table = np.zeros((n_observations,n_actions))
# Each row corresponsds to an observation.
# Each column in that row corresponds to the Q-value of taking that action in that state/observation

# Initialize N table to 0
N_table = np.zeros((n_observations,n_actions), dtype=np.int64)
# Records counts of state-action observations, used for exporation action selection

# alpha / learning rate
a = 0.2

# gamma / discount
g = 0.975



n_episodes = 10000
max_episode_len = 100
all_total_rewards = np.zeros(n_episodes)
epsilon = 1
decay_eps = 0.001
min_eps = 0.001

for e in range(n_episodes):
    if e == 9990:
            env = gym.make("FrozenLake-v1", render_mode="human")
    observation, _ = env.reset() # start new episode
    reward = 0
    last_obs = None
    last_act = None
    
    done = False
    total_reward = 0
    for i in range(max_episode_len):
        # Action selection
        if np.random.uniform() < epsilon:
            action = env.action_space.sample() # random
        else:
            action = np.argmax(Q_table[observation,:]) # best action
        
        # Take action
        new_observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Update Q table
        # From Russel/Norvig
        #Q_table[observation,action] += a * (reward + g*np.max(Q_table[new_observation,:] - Q_table[observation,action]))
        Q_table[observation,action] = (1-a)*Q_table[observation,action] + a * (reward + g*np.max(Q_table[new_observation,:]))

        if terminated or truncated:
            break
        observation = new_observation

    # record total reward for this episode    
    all_total_rewards[e] = total_reward
    epsilon = max(min_eps, np.exp(-decay_eps*e))
    if e > 0 and e % 1000 == 0:
        print(f"{e}/{n_episodes} episodes, eps {epsilon}")
        print(f"{all_total_rewards[e-1000:e].mean()} mean rewards last 1000 episodes")

print(Q_table)

n = int(n_episodes/100)
x = np.arange(n)
y = np.zeros(n)
for i in x:
    y[i] = all_total_rewards[n*i:n*(i+1)].mean()
plt.plot(x,y)
plt.show()
#print(all_total_rewards)
# decay = 0.0001, 30000 episodes, last 1000 mean rewards = 0.494 
# decay = 0.01     =                                       0.000  !!!!!!
# decay = 0.00001     =                                  = 0.023 !
#