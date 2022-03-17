Ongoing RL implementation of a logistic problem. 

Still in a draft state. 

### Introduction

Consider of a list of items characterized by their volumes and their masses. Give a bag of a fixed given volume, the purpose is to maximize the mass of the bag, once filled. 

The approach followed is reinforcement learning. The observation space is the list of items placed in the bag (represented by an array of order (2, number_of_items)), and an action consists in taking one item from the list of items which has not yet been placed in the bag and place it in the bag. Note that this is problematic since it suggests that the action space is dynamic, which is not implemented yet. 

A step consists in picking an item not placed in the bag yet. The output of a step is:
<ul>
<li>State: the content of the bag, i.e. the list of items placed in the bags.</li>
<li>Reward: the reward for the step is the mass of the item added in the bag. </li>
</ul>

The procedure stops, i.e. no further steps can be taken, when:
<ul>
<li>There is no item left.</li>
<li>The bag is not filled, but no other items fit in the bag.</li>
<li>The bag is at maximum capacity.</li>
</ul>

### Usage
<ul>
<li>Requirements can be installed performing

```pip install -r requirements.txt```

</li>
<li>The custom Gym environment can be called as: 

```bag = Logistic(config)```
where 

```config = {"bag_volume": bag_volume, "items": items}``


where `bag_volume` is a float, and `items` is a list of pairs of floats. See run_env_logistic.py for a basic file. </li>
<li>The Python tests of the environments can be run as

 ```pytest -v tests/test_logistic_env.py```
 
</li>
</ul>

### Remark on the action space. 

An action consists in taking an item not yet placed in the bag, and placing in the bag. Hence, the set action is dynamical, which is not yet readily implemented in gym. The list of allowed actions can be accessed as<br />
```
logistic = Logistic()
logistic.allowed_actions()
```
Since the standard method `logistic.action_space.sample()`
does distinguish between allowed action, a new sampling techniques has been introduced as `logistic.items_sampler()`. This should be changed. 






# logistic_rl
