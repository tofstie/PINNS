import tensorflow as tf
import numpy as np
from typing import List, Dict

from src.model.physics_based_nerual_network import PhysicsBasedNerualNetwork

class BurgersNerualNetwork(PhysicsBasedNerualNetwork):
    def __init__(self,
                 model_layers: List[tf.keras.layers.Layer],
                 burgers_parameters: Dict[str, np.ndarray]
                 ):
        super().__init__(model_layers)
        self.burgers_parameters = burgers_parameters
