import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.dataloader.dataloader_base import DataLoaderBase
from src.model.neural_network_base import NeuralNetworkBase
class Pipeline:
    """
    Runs the NN Workflow
    """
    def __init__(self, data_loader: DataLoaderBase, model: NeuralNetworkBase, loss):
        self.data_loader = data_loader
        self.model = model
        self.loss = loss

    def run(self, features: np.ndarray, labels: np.ndarray, compile_params: dict, train_param: dict):
        self.data_loader.load_data(features,labels)
        X_train, Y_train = self.data_loader.get_train_data()
        X_test, Y_test = self.data_loader.get_test_data()
        if train_param.pop("scaler_required"):
            self.loss.set_scaler(self.data_loader.scaler)
        self.model.compile(**compile_params)

        print("\n--- Training Model ---")

        self.model.train(X_train, Y_train, **train_param)

        print("\n--- Evaluating Model ---")
        loss, accuracy = self.model.evaluate(X_test, Y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        print("\n--- Predictions ---")
        predictions = self.model.predict(X_test)
        predicted_classes = predictions
        for i in range(min(5, len(X_test))):
            print(f"Sample {i+1}: Predicted class: {predicted_classes[i]}, Actual class: {Y_test[i]}")
