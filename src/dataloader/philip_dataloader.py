from src.dataloader.dataloader_base import  DataLoaderBase
import numpy as np

class DataLoaderPhilip(DataLoaderBase):

    def __init__(self, test_size: float = 0.2, random_state: int = 1):
        super().__init__(test_size, random_state)

    def load_data(self, features: np.ndarray, labels: np.ndarray):
        return