"""Create the content to store of a bag"""
import numpy as np


class BagContent(list):
    """
    create a list a random of tuple
    """

    def __init__(self, parameters: dict):
        """
        Initiate the class.

        Args:
            parameters: a dictionary containing the following entries:
            "n_items": float
            "max_volume": float
            "max_mass": float
        """
        self.parameters = parameters
        list_parameters = ["n_items", "max_volume", "max_mass"]
        for param in list_parameters:
            assert (
                param in self.parameters.keys()
            ), f"the parameters does not have the right entry: {param} missing"
            assert (
                type(self.parameters[param]) == float
            ), "The type of the entry {param} is not correct"

    def standard_bag(self):
        """Return a 3 * cut_items pairs of of respectively volumes (cut_items, 2*cut_items, 3*cut_items)
        and masses (3* cut_masses, 2*cut_masses, cut_masses).
        """
        cut_items = int(np.floor(self.parameters["n_items"] / 3))
        cut_volumes = int(np.floor(self.parameters["max_volume"] / 3))
        cut_masses = int(np.floor(self.parameters["max_mass"] / 3))

        volumes = (
            [cut_volumes] * cut_items
            + [2 * cut_volumes] * cut_items
            + [3 * cut_volumes] * cut_items
        )
        masses = (
            [3 * cut_masses] * cut_items
            + [2 * cut_masses] * cut_items
            + [cut_masses] * cut_items
        )

        items = list(zip(volumes, masses))

        return items

    def perfect_bag(self, n_cat):
        """The perfect bags returns a bag for which there is a unique well know solutions to the logistic problem.
        Args:
            n_items_cat, int: the number of items per category of weights.
            n_cat, int: the number of categories.

        Return:
            a list of items: a list of length n_items of pair of floats.
            a bag volume: a float setting the volume of the bag for a perfect solution.
        """
        assert (
            self.parameters["n_items"] % n_cat == 0
        ), "the bag cannot be paerfect: n_cat must be a dividor of items."
        volumes = []
        masses = []
        n_items_cat = int(self.parameters["n_items"] / n_cat)
        for i in range(n_cat):
            volumes += [10**i] * n_items_cat
            masses += [10 ** (-i)] * n_items_cat
        items = list(zip(volumes, masses))
        bag_volume = n_items_cat
        return items, bag_volume

    def masses(self, items):
        return [mass for (_, mass) in items]

    def volumes(self, items):
        return [vol for (vol, _) in items]
