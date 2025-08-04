import os

import numpy as np
import tensorflow as tf

from scipy.io import loadmat

from src.pipeline.pipeline import Pipeline
from src.dataloader.dataloader_factory import DataLoaderFactory
from src.model.pendulum_nerual_network import PendulumNeuralNetwork
from src.loss.pendulum_loss import PendulumLoss

if __name__ == "__main__":
    data_path = os.path.join("..","data","PENDULUM.mat")
    pen_data = loadmat(data_path)
    theta = pen_data["theta_improved"]
    t = pen_data["T"].flatten()[:,None]
    k = pen_data["k"].flatten()[:,None]

    K,T = np.meshgrid(k, t)

    X = np.hstack((T.flatten()[:,None], K.flatten()[:,None]))

    Y = theta.flatten()[:,None]
    input_shape = (X.shape[1],)
    model_architecture = [
        tf.keras.layers.Dense(2, activation=tf.nn.relu, input_shape=input_shape),
        tf.keras.layers.Dense(20, activation=tf.nn.relu),
        tf.keras.layers.Dense(20, activation=tf.nn.relu),
        tf.keras.layers.Dense(20, activation=tf.nn.relu),
        tf.keras.layers.Dense(20, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
    ]

    pendulum_parameters = {
        "damping_coefficient": 0.1,
        "residual_factor": 1.0,
        "initial_condition_factor": 1.0
    }

    loss_function = PendulumLoss(pendulum_parameters)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    compile_parameters = {
        "optimizer": "adam",
        "loss": loss_function,
        "metrics": [tf.keras.metrics.Accuracy]
    }

    train_parameters = {
        "epochs": 200,
        "batch_size": 128,
        "validation_split": 0.05,
    }



    dataloader = DataLoaderFactory().create_data_loader("pendulum",test_size=0.2,random_state=42)
    NN = PendulumNeuralNetwork(model_architecture,pendulum_parameters=pendulum_parameters)
    NN.set_loss_function(loss_function)
    loss_function.model = NN
    pipe = Pipeline(dataloader,NN, loss_function)
    pipe.run(
        features=X,
        labels=Y,
        compile_params = compile_parameters,
        train_param=train_parameters
    )