import numpy as np
import tensorflow as tf
from typing import List, Tuple

from keras.src.metrics.accuracy_metrics import accuracy


class NeuralNetworkBase:
    """
    Neural Network Base Class
    """
    def __init__(self, model_layers: List[tf.keras.layers.Layer]):
        self.model = tf.keras.models.Sequential(model_layers)

    def compile(self, optimizer: str, loss: str, metrics: List[str]):
        """Compile the Model"""
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.summary()

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, validation_split: float):
        """Virtual Function to train the neural network"""
        return

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Function to evaluate the neural network"""
        loss,accuracy = self.model.evaluate(x,y,verbose=0)
        return loss, accuracy

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Function to predict"""
        return self.model.predict(x)