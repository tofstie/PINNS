import tensorflow as tf
import numpy as np
from tensorflow.python.eager import tape

from src.loss.physics_informed_loss import PhysicsInformedLoss

class PendulumLoss(PhysicsInformedLoss):
    """Pendulum Loss Class"""
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters,**kwargs)
        self.damping_coefficient = parameters['damping_coefficient']
        self.initial_position = parameters['initial_position']
        self.initial_velocity = parameters['initial_velocity']

    def residual_error(self, x_input, y_pred) -> float:
        """Function to compute the residual error of the Pendulum PDE"""
        u, du_dt, d2u_dt2 = self.compute_gradients(x_input)
        damping_coefficient = tf.cast(tf.multiply(self.scaler_std[1],x_input[:,1]) + self.scaler_mean[1],dtype=tf.float32)
        residual = d2u_dt2 + tf.multiply(damping_coefficient, du_dt) + tf.sin(u)
        return tf.reduce_mean(tf.square(residual))

    def compute_gradients(self, x_input):
        """Function to compute the gradients of the Pendulum PDE using AD"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_input)
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch(x_input)
                u = self.model.model(x_input,training=False)
            du_dt = inner_tape.gradient(u, x_input)
            du_dt = tf.divide(du_dt, self.scaler_std)
        d2u_dt2 = tape.gradient(du_dt, x_input)
        d2u_dt2 = tf.divide(d2u_dt2, tf.square(self.scaler_std))
        return u[:,0], tf.cast(du_dt[:,0],dtype=tf.float32), tf.cast(d2u_dt2[:,0],dtype=tf.float32)

    @tf.function
    def initial_condition_error(self, x_input) -> float:
        shapes =  tf.shape(x_input)
        r = shapes[0]
        c = shapes[1]
        # Create indices to update the second column (index 1).
        indices = tf.stack([tf.range(r, dtype=tf.int32), tf.ones(r, dtype=tf.int32)], axis=1)

        # Get the values from the second column of the original input.
        updates = x_input[:, 1]

        # Create a tensor of zeros with the same shape as the input
        zero = tf.zeros((r, c), dtype=x_input.dtype)
        x_init_scaled = (zero-self.scaler_mean)/self.scaler_std
        # Replace the second column of our scaled tensor.
        x_init = tf.tensor_scatter_nd_update(x_init_scaled, indices, updates)
        u,du_dt,d2u_dt = self.compute_gradients(x_init)
        residual_position = u - tf.constant(self.initial_position,dtype=tf.float32)
        residual_velocity = du_dt - tf.constant(self.initial_velocity,dtype=tf.float32)
        return tf.add(tf.reduce_mean(tf.square(residual_velocity)),tf.reduce_mean(tf.square(residual_position)))