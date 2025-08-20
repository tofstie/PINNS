import tensorflow as tf
import numpy as np
from tensorflow.python.eager import tape

from src.loss.physics_informed_loss import PhysicsInformedLoss

class AdvectionLoss(PhysicsInformedLoss):
    """Linear Advection Loss Class"""
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.initial_position = parameters['initial_position']
        self.boundary_condition = parameters['boundary_condition']
        self.a = parameters['wave_speed']
        self.dim = parameters['dimension']

    def residual_error(self, x_input, y_pred) -> tf.Tensor:
        """Function to compute the residual error of the Linear Advection Equation"""
        u, du_dt, grad_u = self.compute_gradients(x_input)
        residual = du_dt
        for idim in range(self.dim):
            residual += tf.multiply(self.a, grad_u[:, idim])
        return tf.reduce_sum(tf.square(residual))

    def compute_gradients(self, x):
        """Function to compute the gradient needed for the residual error"""
        with tf.GradientTape(persistent=True) as grad_tape:
            grad_tape.watch(x)
            u = self.model.model(x, training=False)
        gradients = grad_tape.gradient(u, x)
        gradients = tf.divide(gradients, self.scaler_std)
        return u, gradients[:,0], gradients[:,1:]

    def initial_condition_error(self, x_input) -> tf.Tensor:
        """Function to compute the initial condition error"""
        shapes = tf.shape(x_input)
        r = shapes[0]
        c = shapes[1]
        ones_tensor = tf.concat([idim * tf.ones(r, dtype=tf.int32) for idim in range(self.dim)], axis = 0)
        index_tensor = tf.concat([tf.range(r, dtype=tf.int32) for idim in range(self.dim)], axis = 0)
        indices = tf.stack([index_tensor, ones_tensor], axis=1)
        positions = tf.reshape(x_input[:,1:],[r*(c-1)])
        zero = tf.zeros((r,c), dtype=x_input.dtype)
        x_init_scaled = (zero-self.scaler_mean)/self.scaler_std

        x_init = tf.tensor_scatter_nd_update(x_init_scaled, indices, positions)
        u = self.model.model(x_init, training=False)
        u = tf.reshape(u, [r])
        if self.initial_position == "cos":
            scaled_input = tf.multiply(x_input[:,1:],self.scaler_std[1]) + self.scaler_mean[1]
            function_input = tf.reduce_sum([scaled_input[:,idim] for idim in range(self.dim)],0)
            residual = u - tf.cos(function_input)
        else:
            residual = tf.constant(0);
        return tf.reduce_sum(tf.square(residual))

    def boundary_condition_error(self, x_input) -> tf.Tensor:
        """Function to compute the boundary condition error of the Linear Advection PDE"""
        if self.boundary_condition == 'periodic':
            return tf.constant(0.0, dtype=tf.float64)
        return tf.constant(0.0, dtype=tf.float64)