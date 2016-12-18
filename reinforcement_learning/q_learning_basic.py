import gym
import numpy as np

env = gym.make('FrozenLake-v0')



#Q table learning algorithm
Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = 0.85
y = .99
num_episodes = 2000

#List for total rewards in steps per episode
rList = []
for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0

    while j < 99:
        j += 1
        #Choose action greedily with noise
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * 1./(i+1))
        #New state and reward from environment
        s1, r, d, _ = env.step(a)
        #Update Q table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r+ y*np.max(Q[s1, :]) - Q[s,a])
        rAll += r
        s = s1

        if d == True:
            break

    rList.append(rAll)


print("Score over time: ", str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)