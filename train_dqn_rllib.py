import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.registry import register_env

from logistic_rl.envs import Bag
from aux.bagcontent import BagContent

# Defining the bag

parameters = {"n_items": 100.0, "max_volume": 100.0, "max_mass": 100.0}
n_cat = 2
bagcontent, bag_volume = BagContent(parameters).perfect_bag(n_cat)
config_env = {"bag_volume": bag_volume, "items": bagcontent}


def env_creator(env_config):
    return Bag(env_config)  # return an env instance


register_env("Bag", env_creator)

# 2 - Creating the agent
ray.init(ignore_reinit_error=True)
# register the environment
# agent's config
config = dqn.DEFAULT_CONFIG.copy()
config["num_workers"] = 1
config["env_config"] = config_env


agents = dqn.DQNTrainer(env=Bag, config=config)
# 3. Training
n_episodes = 10
save_model_freq = 2
save_model_path = "model/"
for i in range(n_episodes):
    training_res = agents.train()
    # printing the min, max and average reward
    print(
        "Episode: {}, Min: {:.2f}, Max: {:.2f}, Average: {:.2f}".format(
            i,
            training_res["episode_reward_min"],
            training_res["episode_reward_max"],
            training_res["episode_reward_mean"],
        )
    )
    # saving the model
    if i % save_model_freq == 0:
        agents.save(save_model_path)
# 4. Saving the model
agents.save()
# 5. Shutting down the workers
ray.shutdown()
print("Done!")
