import numpy as np
from env.logistic_env import Logistic


bag_volume = 100.0
n_items = 10
volumes = np.random.randint(low=1, high=100, size=n_items)
masses = np.random.randint(low=1, high=100, size=n_items)
items = list(zip(volumes, masses))

bag = Logistic(bag_volume=bag_volume, items=items)


