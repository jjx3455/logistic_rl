"""Class defining the gym env for the logistic problem"""
from typing import List
import copy
import numpy as np
import gym
from gym import spaces


class Logistic(gym.Env):
    """
    Gym class for the logistic problems.
    Args:
        bag_volume: the max volume of the bag.
        items : a list of pairs of real numbers (volume, mass).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, bag_volume: float = 100.0, items: List = []):
        super(Logistic, self).__init__()

        self.items = copy.deepcopy(items)
        self.bag_volume = bag_volume
        # The state of the env is the content of the bag. This initializes the state.
        self.bag_content = np.zeros((len(self.items), 2))
        self.remaining_items = copy.deepcopy(self.items)
        self.packed_volume = np.sum(self.bag_content[:, 0])
        self.packed_mass = np.sum(self.bag_content[:, 1])

        # Parameters for the reinforcement learning.
        self.action_space = spaces.Discrete(len(self.items))
        self.allowed_actions = list(range(len(self.items)))
        low = np.zeros((len(self.items), 2))
        a = self.bag_volume * np.ones((len(self.items), 1))
        b = np.inf * np.ones((len(self.items), 1))
        high = np.concatenate((a, b), axis=1)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(len(self.items), 2), dtype=np.float64
        )
        # Keeping tracks of the number of items packed
        self.n_items_packed = 0

    def reset(self, initial_state=None):
        self.bag_content = []
        print("Bag emptied")
        return self.bag_content

    def step(self, action):
        """
        Take one items and place it in the bag.

        Args:
            The action consists in picking one item in the list of items and place it in the bag.

        Returns:
            state: the content of the bag, as (2, n_items) np array.
            reward: the reward of the action, ie the mass of the added item
            done: a bool, If True, the bag cannot be filled any longer.
            {}: Info logistic dictionary.
        """
        # check if the item exists.
        assert action in self.allowed_actions, "The action is not allowed."
        # Pick the item
        item = self.items[action]
        # Check if the item fits in the bag
        if self.packed_volume + item[0] > self.bag_volume:
            print("The item does not fit in the bag.")
            # I do not modify the bag content
            state = self.bag_content
            # Hence no reward
            reward = 0
        # if it fits, put it in the bag
        else:
            # I am adding the content to the bag.
            self.bag_content[self.n_items_packed, :] = item
            self.n_items_packed += 1
            self.packed_volume = np.sum(self.bag_content[:, 0])
            self.packed_mass = np.sum(self.bag_content[:, 1])
            state = self.bag_content
            # The reward for the step is the mass of the item I have added
            reward = item[1]
            # I am deleting the bag from the list of remaining items.
            self.remaining_items.remove(item)
            self.allowed_actions.remove(action)
        # Check if they are still items to be added:
        if self.remaining_items == []:
            print("No more items to add.")
            done = True
        else:
            # Case when the bag is at maximum capacity
            if self.packed_volume == self.bag_volume:
                done = True
            else:
                # We now check if the bag can be filled further.
                min_val_remaining_item = min(
                    [vol for (vol, _) in self.remaining_items if vol != 0]
                )
                # Check if we can add another item to the bag.
                if self.packed_volume + min_val_remaining_item > self.bag_volume:
                    print("The bag is full, no other item can be added")
                    done = True
                else:
                    done = False

        return state, reward, done, {}

    def render(self, mode="human"):
        raise NotImplementedError

    def items_sampler(self):
        """Function which replaces the action.sample() since we do not to repeat the action state."""
        assert (
            self.allowed_actions != []
        ), "the list of allowed action should not be empty."
        return np.random.choice(self.allowed_actions)
