from typing import Tuple

import numpy as np

from sklearn.preprocessing import StandardScaler


class DataLoaderBase:
    """
    Handles loading, splitting, and preparing data
    """
    def __init__(self, test_size: float, random_state: int):
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.Y_test = None
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def load_data(self, features: np.ndarray, labels: np.ndarray):
        """
        Virtual Function to load data
        :return:
        """
        return

    def get_train_data(self):
        return self.X_train, self.Y_train

    def get_valid_data(self):
        return self.X_valid, self.Y_valid

    def get_test_data(self):
        return self.X_test, self.Y_test

    def get_input_size(self):
        return self.X_train.shape[1]

