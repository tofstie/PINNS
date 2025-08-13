import tensorflow as tf
import numpy as np
from tensorflow.python.eager import tape

from src.loss.physics_informed_loss import PhysicsInformedLoss

class BurgersLoss(PhysicsInformedLoss):
    """Burgers Loss Class"""
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters,**kwargs)
        self.initial_position = parameters['initial_position']
        self.boundary_condition = parameters['boundary_condition']

    def residual_error(self, x_input, y_pred) -> tf.Tensor:
        """Function to compute the residual error of the Burgers PDE"""
        shapes = tf.shape(x_input)
        u, du_dt, du_dx = self.compute_gradients(x_input)
        u = tf.reshape(u, [shapes[0]])
        residual = du_dt + tf.multiply(u,du_dx)
        return tf.reduce_sum(tf.square(residual))

    def compute_gradients(self, x):
        with tf.GradientTape(persistent=True) as grad_tape:
            grad_tape.watch(x)
            u = self.model.model(x,training=False)
        gradients = grad_tape.gradient(u, x)
        gradients = tf.divide(gradients, self.scaler_std)
        return u, gradients[:,0], gradients[:,1]

    def initial_condition_error(self, x_input) -> tf.Tensor:
        """ Computers the error in the initial condition of the Burgers PDE"""
        shapes = tf.shape(x_input)
        r = shapes[0]
        c = shapes[1]

        indices = tf.stack([tf.range(r, dtype=tf.int32), tf.ones(r, dtype=tf.int32)], axis=1)

        positions = x_input[:,1]

        zero = tf.zeros((r,c), dtype=x_input.dtype)
        x_init_scaled = (zero-self.scaler_mean)/self.scaler_std

        x_init = tf.tensor_scatter_nd_update(x_init_scaled, indices, positions)
        u = self.model.model(x_init,training=False)
        u = tf.reshape(u, [r])
        if self.initial_position == "cos":
            scaled_input = tf.multiply(positions,self.scaler_std[1]) + self.scaler_mean[1]
            residual = u - tf.cos(scaled_input)
        else:
            residual = tf.constant(0.0)
        return tf.reduce_sum(tf.square(residual))

    def boundary_condition_error(self, x_input) -> tf.Tensor:
        """Function to compute the boundary condition error of the Burgers PDE"""
        if self.boundary_condition == 'periodic':
            return tf.constant(0.0, dtype=tf.float64)
        return tf.constant(0.0, dtype=tf.float64)




