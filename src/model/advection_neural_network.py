import tensorflow as tf
import numpy as np
from typing import List, Dict

from src.model.physics_based_nerual_network import PhysicsBasedNerualNetwork

class AdvectionNeuralNetwork(PhysicsBasedNerualNetwork):
    def __init__(self,
                 model_layers: List[tf.keras.layers.Layer],
                 advection_parameters: Dict[str, float]):
        super().__init__(model_layers)
        self.pendulum_parameters = advection_parameters

