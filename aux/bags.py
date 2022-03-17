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

    def masses(self, items):
        return [mass for (_, mass) in items]

    def volumes(self, items):
        return [vol for (vol, _) in items]
