from logistic_rl.envs import Bag
from aux.bags import BagContent

parameters = {"n_items": 30.0, "max_volume": 100.0, "max_mass": 100.0}
bagcontent = BagContent(parameters).standard_bag()
masses = [mass for (_, mass) in bagcontent]
vols = [vol for (vol, _) in bagcontent]
BAG_VOLUME = 300.0
config_env = {"bag_volume": BAG_VOLUME, "items": bagcontent}

bag = Bag(config_env)


