import os

import numpy as np
import matplotlib.pyplot as plt
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

    data_path_sample = os.path.join("..", "data", "PENDULUM_SAMPLE.mat")
    pen_sample_data = loadmat(data_path_sample)
    theta_sample = pen_sample_data["theta_improved"]
    t_sample = pen_sample_data["T"].flatten()[:, None]
    k_sample = pen_sample_data["k_sample"].flatten()[:, None]

    K,T = np.meshgrid(k, t)

    X = np.hstack((T.flatten()[:,None], K.flatten()[:,None]))

    Y = theta.flatten()[:,None]
    input_shape = (X.shape[1],)
    model_architecture = [
        tf.keras.layers.Dense(2, activation=tf.nn.tanh, input_shape=input_shape),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh),
        tf.keras.layers.Dense(1)
    ]

    pendulum_parameters = {
        "damping_coefficient": 0.1,
        "residual_factor": 1.0,
        "initial_condition_factor": 1.0,
        "initial_velocity": 0.0,
        "initial_position": np.pi/4
    }

    loss_function = PendulumLoss(pendulum_parameters)
    learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler)
    compile_parameters = {
        "optimizer": optimizer,
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

    time_axis = np.linspace(0,30,2000)
    k = k_sample*np.ones(len(time_axis))
    inputs = np.zeros((len(time_axis),2))
    inputs[:,0] = time_axis
    inputs[:,1] = k
    inputs = dataloader.scaler.transform(inputs)
    inputs_sample = np.zeros((len(t_sample),2))
    inputs_sample[:,0] = t_sample[:,0]
    inputs_sample[:,1] = k_sample * np.ones((len(t_sample),1))[:,0]
    inputs_sample = dataloader.scaler.transform(inputs_sample)
    output = NN.model(inputs, training=False)
    fig, axs = plt.subplots(2,1)
    axs[0].plot(inputs[:,0],output,label = "PINN")
    axs[0].plot(inputs_sample[:,0],theta_sample,label = "Original Model")
    axs[0].legend()
    axs[1].plot(inputs[:,0],np.abs(output - theta_sample),label = "L2 Error")
    axs[1].set_yscale("log")
    plt.show()
