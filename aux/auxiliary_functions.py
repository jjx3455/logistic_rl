import numpy as np

class AuxiliaryFunctions():

    def __init__(self) -> None:
        pass

    def rolling_average(observation: list, window_size: int = 10):
        chunk = observation[-window_size:-1]
        average = np.mean(chunk)
        observation = observation + [average] * window_size
        len_obs = len(observation)
        list_rolling_average = []
        for ind in range(window_size, len_obs):
            chunk = observation[ind - window_size : ind]
            average = np.mean(chunk)
            list_rolling_average.append(average)
        return list_rolling_average
