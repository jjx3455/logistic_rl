from logistic_rl.envs import Bag
from aux.bags import BagContent
import gym, ray
from ray.rllib.agents import ppo

parameters = {"n_items": 30.0, "max_volume": 100.0, "max_mass": 100.0}
bagcontent = BagContent(parameters).standard_bag()
masses = [mass for (_, mass) in bagcontent]
vols = [vol for (vol, _) in bagcontent]
BAG_VOLUME = 300.0
config_env = {"bag_volume": BAG_VOLUME, "items": bagcontent}

config = {
    "env_config": config_env,
    "num_workers": 2,
    "framework": "tf2",
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "evaluation_num_workers": 1,
           # config to pass to env class
}

ray.init()
trainer = ppo.PPOTrainer(env=Bag, config=config)

for _ in range(3):
    print(trainer.train())

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
trainer.evaluate()