import tensorflow as tf
import numpy as np
class PhysicsInformedLoss:
    """Physics Informed Loss Base Class"""
    def __init__(self, parameters, **kwargs):
        if 'residual_factor' in parameters:
            self.residual_factor = parameters['residual_factor']
        else:
            self.residual_factor = 0.0

        if 'initial_condition_factor' in parameters:
            self.initial_condition_factor = parameters['initial_condition_factor']
        else:
            self.initial_condition_factor = 0.0

        if 'boundary_condition_factor' in parameters:
            self.boundary_condition_factor = parameters['boundary_condition_factor']
        else:
            self.boundary_condition_factor = 0.0

        self.scaler = None
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None


    def call(self, y_true, y_pred, x):
        MSE = tf.reduce_sum(tf.square(y_true - y_pred))
        residual = self.residual_error(x, y_pred)
        initial_conditions_error = self.initial_condition_error(x)
        boundary_error = self.boundary_condition_error(y_pred)
        return MSE + self.initial_condition_factor*initial_conditions_error + self.residual_factor*residual + self.boundary_condition_factor * boundary_error

    def residual_error(self, x_input, y_pred) -> tf.Tensor:
        """Virtual Function to compute the residual error of the PDE"""
        return tf.constant(0.0, dtype=tf.float64)

    def initial_condition_error(self, x_input) -> tf.Tensor:
        """Virtual Function to compute the initial condition error of the PDE"""
        return tf.constant(0.0, dtype=tf.float64)

    def boundary_condition_error(self, x_input) -> tf.Tensor:
        """Virtual Function to compute the boundary condition error of the PDE"""
        return tf.constant(0.0, dtype=tf.float64)

    def set_scaler(self, scaler):
        self.scaler = scaler
        self.scaler_mean = tf.constant(self.scaler.mean_,dtype = tf.float64)
        self.scaler_std = tf.constant(self.scaler.scale_, dtype = tf.float64)
        return