import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
import logging


class Logging:
    def __init__(self, list_record) -> None:

        now = dt.now()

        self.list_record = list_record
        self.record = dict()
        for a_list in self.list_record:
            self.record[a_list] = []

        now = now.strftime("%d_%m_%Y_%H_%M_%S")
        training_data_dir = "training_data"
        path_training_folder = training_data_dir + "/" + now
        if not os.path.exists(training_data_dir):
            os.mkdir(training_data_dir)
        if not os.path.exists(path_training_folder):
            os.mkdir(path_training_folder)

        logging.basicConfig(
            filename=path_training_folder + "log.log",
            filemode="a",
            format="%(asctime)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            level=logging.INFO,
        )

    def logging(self, info: dict):
        for key in info.keys():
            logging.info(key, ":", info[key])
