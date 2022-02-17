"""
Source: https://github.com/MatePocs/gym-basic/blob/main/gym_basic_env_test.ipynb
"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from env.logistic_env import Logistic

# Preparing the items.
bag_volume = 10
n_items = 50
volumes = np.random.randint(low=1, high=np.random.randint(2, 10), size=n_items)
masses = np.random.randint(low=1, high=np.random.randint(2, 50), size=n_items)
print("Average mass:", np.mean(masses))
items = list(zip(volumes, masses))

# Creating the bag.
bag = Logistic(bag_volume=bag_volume, items=items)

# training parameters
num_episodes = 200
max_steps_per_episode = n_items  # but it won't go higher than 1

learning_rate = 0.01
discount_rate = 1

exploration_rate = 10
max_exploration_rate = 50
min_exploration_rate = 0.1

exploration_decay_rate = 0.1  # if we decrease it, will learn slower

q_table = [[np.zeros(bag.observation_space.shape), 0, 0]] * bag.action_space.n

rewards_all_episodes = []
counter = 0
# Q-Learning algorithm
for episode in tqdm(range(num_episodes)):
    state = bag.reset()
    done = False
    rewards_current_episode = 0
    counter_action = 0
    states_list = [a_state for [a_state, _, _] in q_table]
    actions_list = [an_action for [_, an_action, _] in q_table]
    qvalues_list = [q_value for [_, _, q_value] in q_table]
    for step in range(max_steps_per_episode):
        # Exploration -exploitation trade-off
        exploration_rate_threshold = np.random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(qvalues_list)
        else:
            action = bag.items_sampler()
        try:
            new_state, reward, done, info = bag.step(action)
        except:  # bad, bad thing.
            pass
        max_reward_future = np.max([mass for (_, mass) in bag.remaining_items])
        q_value = q_table[counter_action][2]
        q_value = (1 - learning_rate) * q_value + learning_rate * (
            reward + discount_rate * (max_reward_future - q_value)
        )
        q_table[counter_action] = [state, action, q_value]
        rewards_current_episode += reward
        state = new_state
        counter_action += 1
        if done == True:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + (
        max_exploration_rate - min_exploration_rate
    ) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)
# Printing counter


plt.plot(rewards_all_episodes)
plt.savefig("reward.pdf")


# Calculate and print the average reward per 10 episodes
rewards_per_thousand_episodes = np.split(
    np.array(rewards_all_episodes), num_episodes / 10
)
count = 10
print("********** Average  reward per thousand episodes **********\n")

for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 10)))
    count += 10

# Print updated Q-table
print("\n\n********** Q-table **********\n")
# print(q_table)
