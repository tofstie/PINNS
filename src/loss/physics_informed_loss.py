import tensorflow as tf
import numpy as np
class PhysicsInformedLoss:
    """Physics Informed Loss Base Class"""
    def __init__(self, parameters, **kwargs):
        self.residual_factor = parameters['residual_factor']
        self.initial_condition_factor = parameters['initial_condition_factor']
        self.scaler = None
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None


    def call(self, y_true, y_pred, x):
        MSE = tf.keras.losses.MSE(y_true, y_pred)
        residual = self.residual_error(x, y_pred)
        initial_conditions_error = self.initial_condition_error(x)

        return MSE + self.initial_condition_factor*initial_conditions_error + self.residual_factor*residual

    def residual_error(self, x_input, y_pred) -> float:
        """Virtual Function to compute the residual error of the PDE"""
        return

    def initial_condition_error(self, x_input) -> float:
        """Virtual Function to compute the initial condition error of the PDE"""
        return

    def boundary_condition_error(self, x_input) -> float:
        """Virtual Function to compute the boundary condition error of the PDE"""
        return

    def set_scaler(self, scaler):
        self.scaler = scaler
        self.scaler_mean = tf.constant(self.scaler.mean_,dtype = tf.float64)
        self.scaler_std = tf.constant(self.scaler.scale_, dtype = tf.float64)
        return