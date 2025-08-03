import tensorflow as tf
import numpy as np

from src.loss.physics_informed_loss import PhysicsInformedLoss

class PendulumLoss(PhysicsInformedLoss):
    """Pendulum Loss Class"""
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters,**kwargs)
        self.damping_coefficient = parameters['damping_coefficient']

    def residual_error(self, x_input, y_pred) -> float:
        """Function to compute the residual error of the Pendulum PDE"""
        u, du_dt, d2u_dt2 = self.compute_gradients(x_input,y_pred)
        residual = d2u_dt2 + self.damping_coefficient * du_dt + tf.sin(u)
        return tf.reduce_mean(tf.square(residual))

    def compute_gradients(self, x_input, y_pred):
        """Function to compute the gradients of the Pendulum PDE using AD"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_input)
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch(x_input)
                u = self.model.model(x_input,training=True)
            du_dt = inner_tape.gradient(u, x_input)
        d2u_dt2 = tape.gradient(du_dt, x_input)
        return u[:,0], tf.cast(du_dt[:,0],dtype=tf.float32), tf.cast(d2u_dt2[:,0],dtype=tf.float32)

