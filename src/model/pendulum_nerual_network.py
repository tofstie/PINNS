import tensorflow as tf
import numpy as np
from typing import List, Dict

from src.model.physics_based_nerual_network import PhysicsBasedNerualNetwork

class PendulumNeuralNetwork(PhysicsBasedNerualNetwork):
    def __init__(self,
                 model_layers: List[tf.keras.layers.Layer],
                 pendulum_parameters: Dict[str, float]):
        super().__init__(model_layers)
        self.initial_conditions = None
        self.pendulum_parameters = pendulum_parameters
        self.damping_factor = self.pendulum_parameters['damping_coefficient']

    def set_initial_conditions(self, initial_conditions: np.ndarray):
        self.initial_conditions = initial_conditions
        return
