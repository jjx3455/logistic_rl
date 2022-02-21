from asyncio import exceptions
import os
from datetime import datetime as dt
from tkinter import BASELINE
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from env.logistic_env import Logistic

BAG_VOLUME = 250
N_ITEMS = 750
MAX_VOLUME = 100
MAX_MASS = 100
volumes = (
    [1] * int(np.floor(N_ITEMS / 3))
    + [10] * int(np.floor(N_ITEMS / 3))
    + [100] * int(np.floor(N_ITEMS / 3))
)
masses = (
    [100] * int(np.floor(N_ITEMS / 3))
    + [10] * int(np.floor(N_ITEMS / 3))
    + [1] * int(np.floor(N_ITEMS / 3))
)
items = list(zip(volumes, masses))
BASELINE = np.mean(masses) * BAG_VOLUME / np.mean(volumes)
MAX_BASELINE = 25000
remark = "Test run."

bag = Logistic(bag_volume=BAG_VOLUME, items=items)


num_episodes = 2000
max_steps_per_episode = N_ITEMS  # but it won't go higher than 1


def rolling_average(observation: list, window_size: int = 10):
    chunk = observation[-window_size:-1]
    average = np.mean(chunk)
    observation = observation + [average] * window_size
    len_obs = len(observation)
    list_rolling_average = []
    for ind in range(window_size, len_obs):
        chunk = observation[ind - window_size : ind]
        average = np.mean(chunk)
        list_rolling_average.append(average)
    return list_rolling_average


now = dt.now()
now = now.strftime("%d_%m_%Y_%H_%M_%S")
training_data_dir = "training_data"
path_training_folder = training_data_dir + "/" + now

if not os.path.exists(training_data_dir):
    os.mkdir(training_data_dir)

if not os.path.exists(path_training_folder):
    os.mkdir(path_training_folder)


max_exploration_rate = 1
min_exploration_rate = 0.1
exploration_decay_rate = 0.01
# seeding the exploration rate
exploration_rate = 1


learning_rate = 0.1
discount_rate = 1
# if we decrease it, will learn slower
q_table = [[np.zeros(bag.observation_space.shape), 0, 0]] * bag.action_space.n
counter = 0
n_items_bag = []
filled_proportion = []
average_vol_item = []
average_mass_items = []
total_mass_packed = []
mean_q_value = []
sampler_counts = []
take_best_q_count = []
exceptions_counts = []
# Q-Learning algorithm
for episode in tqdm(range(num_episodes), leave=False, desc="episode"):
    state = bag.reset()
    done = False
    rewards_current_episode = 0
    # counter_action = 0
    states_list = [a_state for [a_state, _, _] in q_table]
    actions_list = [an_action for [_, an_action, _] in q_table]
    qvalues_list = [q_value for [_, _, q_value] in q_table]
    count_sampler = 0
    counter_take_best_q = 0
    count_exception = 0
    for step in range(max_steps_per_episode):
        # Exploration -exploitation trade-off
        exploration_rate_threshold = np.random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            # there must be a btter way.
            qvalues_list_allowed = [qvalues_list[i] for i in bag.allowed_actions]
            action = np.argmax(qvalues_list_allowed)
            action = bag.allowed_actions[action]
            counter_take_best_q += 1
        else:
            action = bag.items_sampler()
            count_sampler += 1
        try:
            new_state, reward, done, info = bag.step(action)
            max_reward_future = np.max([mass for (_, mass) in bag.remaining_items])
            q_value = q_table[bag.n_items_packed][2]
            q_value = q_value + learning_rate * (
                reward + discount_rate * max_reward_future
            )
            q_table[bag.n_items_packed] = [state, action, q_value]
            rewards_current_episode += reward
            state = new_state
        except:  # bad, bad thing.
            count_exception += 1
            pass
        if done == True:
            assert (
                rewards_current_episode == bag.packed_mass
            ), "The accumulated reward does not fit the mass placed in the bag"
            # print("number of actions:", counter_action, "number of items packed:", bag.n_items_packed)
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + (
        max_exploration_rate - min_exploration_rate
    ) * np.exp(-exploration_decay_rate * episode)
    dict_info = bag.infos()
    n_items_bag.append(dict_info["n_items"])
    filled_proportion.append(dict_info["filled"])
    states_list = [a_state for [a_state, _, _] in q_table]
    average_vol = np.mean(states_list[bag.n_items_packed][0 : bag.n_items_packed, 0])
    average_mass = np.mean(states_list[bag.n_items_packed][0 : bag.n_items_packed, 1])
    average_vol_item.append(average_vol)
    average_mass_items.append(average_mass)
    total_mass_packed.append(bag.packed_mass)
    qvalues_list = [q_value for [_, _, q_value] in q_table]
    mean_q = np.mean(qvalues_list)
    mean_q_value.append(mean_q)
    take_best_q_count.append(counter_take_best_q)
    sampler_counts.append(count_sampler)
    exceptions_counts.append(count_exception)


window_average = 15
roll_average_mass = rolling_average(total_mass_packed, window_average)

N_fig = 8

plt.figure(figsize=(8 * 2, N_fig * 5))

plt.figtext(0.15, 0.87, remark, fontsize="large")

text_items = (
    f"bag_volume = {BAG_VOLUME},\n"
    f"n_items = {N_ITEMS},\n"
    f"max_volume = {MAX_VOLUME},\n"
    f"max mass = {MAX_MASS}\n"
    f"Average volumes: = {np.mean(volumes)},\n"
    f"Average mass = {np.mean(masses)},\n"
    f"Total volume = {np.sum(volumes)},\n"
    f"Total mass = {np.sum(masses)},\n"
    f"baseline = {BASELINE} "
)

plt.figtext(0.15, 0.80, text_items, fontsize="large")

text_training = (
    f"num_episodes = {num_episodes},\n"
    f"max_steps_per_episode = {max_steps_per_episode},\n"
    f"learning_rate = {learning_rate},\n"
    f"discount_rate = {discount_rate},\n"
    f"max_exploration_rate = {max_exploration_rate},\n"
    f"min_exploration_rate = {min_exploration_rate},\n"
    f"exploration_decay_rate =  {exploration_decay_rate}"
)
plt.figtext(0.55, 0.80, text_training, fontsize="large")

plt.subplot(N_fig, 2, 3)
plt.hist(volumes, bins=3)
plt.title("Repartion of the volumes of the items")

plt.subplot(N_fig, 2, 4)
plt.hist(masses, bins=3)
plt.title("Repartion of the masses of the items")

plt.subplot(N_fig, 2, 5)
plt.scatter(volumes, masses)
plt.title("Repartion of the masses/volumes")
plt.xlabel("volumes")
plt.ylabel("masses")

plt.subplot(N_fig, 2, 6)
plt.plot(n_items_bag)
plt.title("n_items in bag / episode")
plt.xlabel("episodes")

plt.subplot(N_fig, 2, 7)
plt.plot(roll_average_mass)
x = np.arange(num_episodes)
y = BASELINE * np.ones(num_episodes)
z = MAX_BASELINE * np.ones(num_episodes)
plt.plot(x, y, color="red")
plt.plot(x, z, color="green")
plt.title("Rolling average total mass / episode")
plt.xlabel("episodes")

plt.subplot(N_fig, 2, 8)
plt.plot(average_mass_items)
plt.title("items average mass in bag / episode")
plt.xlabel("episodes")

plt.subplot(N_fig, 2, 9)
plt.plot(total_mass_packed)
plt.plot(x, y, color="red")
plt.plot(x, z, color="green")
plt.title("Total mass in bag / episode")
plt.xlabel("episodes")

plt.subplot(N_fig, 2, 10)
plt.plot(filled_proportion)
plt.title("Proportion vol. filled bag / episode")
plt.xlabel("episodes")

plt.subplot(N_fig, 2, 11)
plt.plot(average_vol_item)
plt.title("items average volume in bag / episode")
plt.xlabel("episodes")

plt.subplot(N_fig, 2, 12)
plt.plot(mean_q_value)
plt.title("mean q value / episode")
plt.xlabel("episodes")

plt.subplot(N_fig, 2, 13)
plt.plot(sampler_counts)
plt.title("count sampler call / episode")
plt.xlabel("episodes")

plt.subplot(N_fig, 2, 14)
plt.plot(take_best_q_count)
plt.title("best q call / episode")
plt.xlabel("episodes")

plt.subplot(N_fig, 2, 15)
plt.plot(exceptions_counts)
plt.title("proportion exception call on step / episode")
plt.xlabel("episodes")

now = dt.now()
now = now.strftime("%d_%m_%Y_%H_%M_%S")

path_to_training_data = path_training_folder + "/" + now + ".pdf"

plt.savefig(path_to_training_data)
plt.close()
