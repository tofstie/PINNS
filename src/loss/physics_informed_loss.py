import tensorflow as tf
import numpy as np
class PhysicsInformedLoss:
    """Physics Informed Loss Base Class"""
    def __init__(self, parameters, **kwargs):
        self.residual_factor = parameters['residual_factor']
        self.initial_condition_factor = parameters['initial_condition_factor']
        self.model = None


    def call(self, y_true, y_pred, x):
        MSE = tf.keras.losses.MSE(y_true, y_pred)
        residual = self.residual_error(x, y_pred)
        return MSE + self.residual_factor*residual

    def residual_error(self, x_input, y_pred) -> float:
        """Virtual Function to compute the residual error of the PDE"""
        return
