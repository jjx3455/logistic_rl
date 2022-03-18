from logistic_rl.envs import Bag
from aux.bagcontent import BagContent
import ray
from ray.rllib.agents import ppo

parameters = {"n_items": 100.0, "max_volume": 100.0, "max_mass": 100.0}
n_cat = 2
bagcontent, bag_volume = BagContent(parameters).perfect_bag(n_cat)
config_env = {"bag_volume": bag_volume, "items": bagcontent}

config = {
    "env_config": config_env,
    "num_workers": 10,
    "framework": "tf2",
    "model": {
        "fcnet_hiddens": [256, 128, 64, 32],
        "fcnet_activation": "relu",
    },
    "evaluation_num_workers": 2,
    # config to pass to env class
}

ray.init()
trainer = ppo.PPOTrainer(env=Bag, config=config)
max_reward = []
hist_reward = []
for _ in range(1):
    result = trainer.train()
    max_reward.append(result["episode_reward_max"])
    max_reward.append(result["hist_stats"]["episode_reward"])


# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
print(trainer.evaluate())
