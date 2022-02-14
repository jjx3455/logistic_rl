"""Script containing the test """
import pytest
import numpy as np
from env.logistic_env import Logistic


class TestLogictic:
    def test_setup(self):
        bag = Logistic()
        assert bag.bag_volume == 100, "The default bag volume is incorrect"
        assert bag.items == [], "The defaults items list is not correct"
        bag_volume = np.random.randint(1, 100)
        n_items = np.random.randint(1, 100)
        volumes = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        assert (
            bag.action_space.n == n_items
        ), "The number of actions does not fit the number of items."
        assert bag.allowed_actions == list(
            range(n_items)
        ), "The number of allowed actions does not fit the number of items."

    def test_reset(self):
        bag_volume = np.random.randint(1, 100)
        n_items = np.random.randint(1, 100)
        volumes = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        to_compare = np.ones((n_items, 2))
        to_check = bag.bag_content == to_compare
        assert to_check.all, "The bag is not empy initially"
        for _ in range(int(np.floor(n_items / 2))):
            try:
                bag.step(bag.action_space.sample())
            except:
                pass
        bag.reset()
        assert bag.bag_content == [], "The bag is not empty after resetting."

    def test_render(self):
        bag = Logistic()
        with pytest.raises(NotImplementedError):
            bag.render()

    def test_step_bag_too_small(self):
        bag_volume = np.random.randint(1, 100)
        n_items = np.random.randint(1, 100)
        volumes = np.random.randint(
            low=bag_volume + 1,
            high=np.random.randint(bag_volume + 2, 1000),
            size=n_items,
        )
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        action = bag.action_space.sample()
        state, reward, done, infos = bag.step(action)
        to_check = state == np.ones((n_items, 2))
        assert (
            to_check.all
        ), "The items is too big for the bag, and should not be in the bag."
        assert reward == 0, "The bag must empty and should not contain any objects."
        assert infos == {}, "The infos should be empty"

    def test_bag_size_enough_for_one_item(self):
        bag_volume = np.random.randint(2, 100)
        n_items = np.random.randint(1, 100)
        volumes = np.random.randint(low=1, high=bag_volume, size=n_items)
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        action = bag.action_space.sample()
        state, reward, done, infos = bag.step(action)
        to_compare = np.ones((n_items, 2))
        to_compare[0, :] = items[action]
        to_check = state == to_compare
        assert to_check.all, "The bag does not contain the right item."
        assert (
            reward == items[action][1]
        ), "The reward is not the mass of the added item."
        assert infos == {}, "The infos should be empty"

    def test_action_not_in_range(self):
        bag_volume = np.random.randint(2, 100)
        n_items = np.random.randint(1, 100)
        volumes = np.random.randint(low=1, high=bag_volume, size=n_items)
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        forbidden_action = n_items + np.random.randint(1, 100)
        with pytest.raises(Exception):
            assert bag.step(forbidden_action)

    def test_allowed_actions(self):
        bag_volume = np.random.randint(2, 100)
        n_items = np.random.randint(1, 100)
        volumes = np.random.randint(low=1, high=bag_volume, size=n_items)
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        action = bag.action_space.sample()
        bag.step(action)
        assert (
            action not in bag.allowed_actions
        ), "The action should have been removed from the allowed actions."
        with pytest.raises(Exception):
            assert bag.step(
                action
            ), "An action cannot be performed twice in a row without raising an exception."

    def test_remaining_items(self):
        # Beware that items can appear in more than one copy in the list of items.
        bag_volume = np.random.randint(2, 100)
        n_items = np.random.randint(1, 100)
        volumes = np.random.randint(low=1, high=bag_volume, size=n_items)
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        action = bag.action_space.sample()
        count_before = items.count(items[action])
        bag.step(action)
        count_after = bag.remaining_items.count(items[action])
        assert (
            count_before - count_after == 1
        ), "The item should have been removed from the bag."

    def test_no_items_left(self):
        bag_volume = np.random.randint(2, 100)
        n_items = 1
        volumes = np.random.randint(low=1, high=bag_volume, size=n_items)
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        action = bag.action_space.sample()
        _, _, done, _ = bag.step(action)
        assert (
            items[action] not in bag.remaining_items
        ), "The item should have been removed from the bag."
        assert bag.remaining_items == [], "The list of remaining items should be empty."
        assert done == True, "There is no more item to add to the bag."

    def test_bag_full_with_one_item(self):
        bag_volume = np.random.randint(2, 100)
        n_items = np.random.randint(1, 100)
        volumes = np.random.randint(low=bag_volume, high=bag_volume + 1, size=n_items)
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        action = bag.action_space.sample()
        _, _, done, _ = bag.step(action)
        assert bag_volume <= bag.packed_volume, "The packed volume should be maximal."
        assert done == True, "The bag should be full."

    def test_bag_cannot_be_filled_with_more_than_one_item(self):
        bag_volume = np.random.randint(3, 100)
        n_items = np.random.randint(1, 100)
        volumes = np.random.randint(
            low=int(np.floor(bag_volume / 2)) + 1,
            high=int(np.floor(bag_volume / 2)) + 2,
            size=n_items,
        )
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        action = bag.action_space.sample()
        print(items[action])
        _, _, done, _ = bag.step(action)
        assert (
            bag_volume > bag.packed_volume
        ), "The packed volume should not be maximal."
        assert (
            done == True
        ), "The bag cannot be filled further, though it is half-empty."

    def test_items_sampler_empty_allowed_actions(self):
        bag_volume = np.random.randint(2, 100)
        n_items = 1
        volumes = np.random.randint(low=1, high=bag_volume, size=n_items)
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        action = bag.action_space.sample()
        bag.step(action)
        with pytest.raises(Exception):
            assert bag.items_sampler()

    def test_items_sampler_non_empty_allowed_actions(self):
        bag_volume = np.random.randint(2, 100)
        n_items = np.random.randint(2, 100)
        volumes = np.random.randint(
            low=1, high=2 + int(np.floor(bag_volume / 4)), size=n_items
        )
        masses = np.random.randint(low=1, high=np.random.randint(2, 100), size=n_items)
        items = list(zip(volumes, masses))
        bag = Logistic(bag_volume=bag_volume, items=items)
        action = bag.action_space.sample()
        bag.step(action)
        assert (
            bag.allowed_actions != []
        ), "The list of allowed actions should not be empty."
        assert (
            bag.items_sampler() in bag.allowed_actions
        ), "The picked action should be in the list of allowed actions."
