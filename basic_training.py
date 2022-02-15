"""
Source: https://github.com/MatePocs/gym-basic/blob/main/gym_basic_env_test.ipynb
"""

import numpy as np
from env.logistic_env import Logistic

# Preparing the items.
bag_volume = 1000
n_items = 300
volumes = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
items = list(zip(volumes, masses))

# Creating the bag.
bag = Logistic(bag_volume=bag_volume, items=items)

# training parameters
num_episodes = 1000
max_steps_per_episode = n_items  # but it won't go higher than 1

learning_rate = 0.01
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01

exploration_decay_rate = 0.01  # if we decrease it, will learn slower

q_table = [[np.zeros(bag.observation_space.shape), 0, 0]] * bag.action_space.n

rewards_all_episodes = []

# Q-Learning algorithm
for episode in range(num_episodes):
    state = bag.reset()
    done = False
    rewards_current_episode = 0
    counter_action = 0
    new_q_table = [[np.zeros(bag.observation_space.shape), 0, 0]] * bag.action_space.n
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
        except:
            pass
        list_test_state = [(new_state == a_state).all() for [a_state, _, _] in q_table]
        if sum(list_test_state):
            # Update Q-table for Q(s,a)
            index_new_state = list_test_state.index(True)
            q_value_new_state = qvalues_list[index_new_state]
            q_value = (1 - learning_rate) * q_value + learning_rate * (
                reward + discount_rate * q_value_new_state
            )
        else:
            q_value = reward

        rewards_current_episode += reward
        state = new_state
        new_q_table[counter_action] = [state, action, q_value]
        counter_action += 1
        if done == True:
            break

    q_table = new_q_table

    # Exploration rate decay
    exploration_rate = min_exploration_rate + (
        max_exploration_rate - min_exploration_rate
    ) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per 10 episodes
rewards_per_thousand_episodes = np.split(
    np.array(rewards_all_episodes), num_episodes / 100
)
count = 100
print("********** Average  reward per thousand episodes **********\n")

for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 100)))
    count += 100

# Print updated Q-table
print("\n\n********** Q-table **********\n")
#print(q_table)
