from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np

from src.dataloader.dataloader_base import  DataLoaderBase


class DataLoaderPendulum(DataLoaderBase):

    def __init__(self, test_size: float = 0.2, random_state: int = 1):
        super().__init__(test_size, random_state)

    def load_data(self, features: np.ndarray, labels: np.ndarray):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            features, labels, test_size=self.test_size, random_state=self.random_state
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return