import tensorflow as tf
import numpy as np
from typing import List

from src.model.neural_network_base import NeuralNetworkBase

class CategoryNeuralNetwork(NeuralNetworkBase):
    def __init__(self, model_layers: List[tf.keras.layers.Layer]):
        super().__init__(model_layers)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, validation_split: float):
        """Function to train the neural network"""
        return self.model.fit(
            x,y,epochs=epochs,batch_size=batch_size,validation_split=validation_split,verbose=1
        )