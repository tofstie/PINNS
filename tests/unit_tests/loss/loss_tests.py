import unittest

import numpy as np
import tensorflow as tf

from src.loss.pendulum_loss import PendulumLoss

class TestPendulumLoss(unittest.TestCase):
    def setUp(self):
        """
        Method that runs before every test
        """
        x = np.array([0,1],[0.1,0.1])
        y = np.array([1,2])
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)
        self.y = tf.convert_to_tensor(y, dtype=tf.float32)
        self.parameters = {
        "residual_factor": 1.0,
        "initial_condition_factor": 1.0,
        "initial_velocity": 0.0,
        "initial_position": np.pi/4
        }
        self.loss = PendulumLoss(self.parameters)
        self.loss.scaler_mean = 0.0
        self.loss.scaler_std = 1.0


    def test_residual_loss(self):
        """
        Test the residual of the loss function
        """
        pass

    def test_initial_condition_loss(self):
        """
        Test the initial condition of the loss function
        """
        pass

    def test_AD(self):
        """
        Tests the automatic difference of the loss function
        """
        pass