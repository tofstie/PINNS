import tensorflow as tf
import numpy as np
from typing import List

from src.model.neural_network_base import NeuralNetworkBase

class PhysicsBasedNerualNetwork(NeuralNetworkBase):
    def __init__(self, model_layers: List[tf.keras.layers.Layer]):
        super().__init__(model_layers)
        self.loss = None
        self.train_metrics = tf.keras.metrics.Mean()
        self.test_metrics = tf.keras.metrics.MeanSquaredError()
        self.val_metrics = tf.keras.metrics.Mean()

    def set_loss_function(self, loss_function):
        self.loss = loss_function

    @tf.function
    def _train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            tape.watch(x_batch)
            y_pred = self.model(x_batch, training=True)
            loss = self.loss.call(y_batch, y_pred, x_batch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_metrics.update_state(loss)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, validation_split: float):
        """Function to train the neural network"""
        if not (0 < validation_split < 1):
            raise ValueError("validation_split must be between 0 and 1.")

        split_at = int(len(x) * (1 - validation_split))
        x_train, x_val = x[:split_at], x[split_at:]
        y_train, y_val = y[:split_at], y[split_at:]

        print(f"Training on {len(x_train)} samples, validating on {len(x_val)} samples.")

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        for epoch in range(epochs):
            self.train_metrics.reset_state()
            self.val_metrics.reset_state()
            for step, (x_batch, y_batch) in enumerate(dataset):
                self._train_step(x_batch, y_batch)
            for x_val_batch, y_val_batch in val_dataset:
                y_val_pred = self.model(x_val_batch, training=False)
                val_loss = self.loss.call(y_val_batch, y_val_pred, x_val_batch)
                self.val_metrics.update_state(val_loss)
            train_loss_result = self.train_metrics.result()
            val_loss_result = self.val_metrics.result()
            print(
                f"Epoch {epoch + 1}, "
                f"Train Loss: {train_loss_result.numpy():.6f}, "
                f"Validation Loss: {val_loss_result.numpy():.6f}"
            )
        return {"final_train_loss": train_loss_result, "final_validation_loss": val_loss_result}

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        x_tensor = tf.convert_to_tensor(x)
        y_tensor = tf.convert_to_tensor(y)
        y_pred = self.model(x_tensor, training=False)
        loss = self.loss.call(y_tensor, y_pred, x_tensor)
        accuracy = self.test_metrics(y_tensor,y_pred)
        return tf.reduce_sum(loss), accuracy