import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.io import loadmat

from src.pipeline.pipeline import Pipeline
from src.dataloader.dataloader_factory import DataLoaderFactory
from src.model.pendulum_nerual_network import PendulumNeuralNetwork
from src.loss.pendulum_loss import PendulumLoss

from src.visualization.visualization_parameters import VisualizationParameters, FigureParameters, DataParameters
from src.visualization.visualization_base import VisualizationBase

if __name__ == "__main__":
    ## Load training and test data
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
    ## Model Parameters
    input_shape = (X.shape[1],)
    model_architecture = [
        tf.keras.layers.Dense(2, activation=tf.nn.tanh, input_shape=input_shape, dtype = tf.float64),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh, dtype = tf.float64),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh, dtype = tf.float64),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh, dtype = tf.float64),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh, dtype = tf.float64),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh, dtype=tf.float64),
        tf.keras.layers.Dense(20, activation=tf.nn.tanh, dtype=tf.float64),
        tf.keras.layers.Dense(1, dtype = tf.float64)
    ]
    pendulum_parameters = {
        "residual_factor": 0.1,
        "initial_condition_factor": 1.0,
        "initial_velocity": 0.0,
        "initial_position": 2*np.pi/5
    }
    train_parameters = {
        "epochs": 800,
        "batch_size": 64,
        "validation_split": 0.05,
        "scaler_required": True
    }

    ## Loss and Optimizer
    loss_function = PendulumLoss(pendulum_parameters)
    learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler)
    compile_parameters = {
        "optimizer": optimizer,
        "loss": loss_function,
        "metrics": [tf.keras.metrics.Accuracy]
    }
    ## Dataloader for train and test data
    dataloader = DataLoaderFactory().create_data_loader("pendulum",test_size=0.2,random_state=42)
    ## Build NN
    NN = PendulumNeuralNetwork(model_architecture,pendulum_parameters=pendulum_parameters)
    NN.set_loss_function(loss_function)
    loss_function.model = NN
    ## Create Pipeline
    pipe = Pipeline(dataloader,NN, loss_function)
    ## Run NN training and testing
    pipe.run(
        features=X,
        labels=Y,
        compile_params = compile_parameters,
        train_param=train_parameters
    )

    ## Loss figure
    # loss_fig_params = FigureParameters()
    # loss_fig_params.type = "figure"
    # loss_fig_params.add_x_axes_settings("Iteration")
    # loss_fig_params.add_y_axes_settings("Loss", "log")
    # loss_data = DataParameters(x = NN.iter_loss[0],y = NN.iter_loss[1], plot_type="plot", label="Loss")
    # loss_fig_params.add_data(loss_data,0)
    fig, ax = plt.subplots(1,1)
    ax.plot(NN.iter_loss[:,0],NN.iter_loss[:,1],label = "Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")

    ## Parameter space figure
    parameter_fig_params = FigureParameters()
    parameter_fig_params.type = "figure"
    parameter_fig_params.add_x_axes_settings("Theta")
    parameter_fig_params.add_y_axes_settings("k")
    x_train, y_train = dataloader.get_train_data()
    x_train = dataloader.scaler.inverse_transform(x_train)
    parameter_data = DataParameters(x = x_train[:,0],y = x_train[:,1], marker = "o", plot_type="scatter")
    parameter_fig_params.add_data(parameter_data,0)

    ## Get data for visualization
    time_axis = np.linspace(0,5,200)
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

    ## Setting up Output Figure
    params = VisualizationParameters()
    output_fig_params = FigureParameters()
    output_fig_params.type = "subplot"
    output_fig_params.subplot_rows = 2
    output_fig_params.subplot_cols = 1
    output_fig_params.add_x_axes_settings("Time (s)")
    output_fig_params.add_y_axes_settings(r'theta', yscale = "linear")
    output_fig_params.add_y_axes_settings(r'L1 error', yscale = "log")
    ## Setting up data
    PINN_data = DataParameters(x = inputs[:,0],y = output,label = "PINN")
    Actual_data = DataParameters(x = inputs_sample[:,0],y = theta_sample,label = "Original Model")
    Error_data = DataParameters(x = inputs[:,0],y = np.abs(output - theta_sample),label = "L2 Error")

    output_fig_params.add_data(PINN_data,0)
    output_fig_params.add_data(Actual_data,0)
    output_fig_params.add_data(Error_data,1)

    params.figures.append(output_fig_params)
    params.figures.append(parameter_fig_params)
    ## Visualize data
    visualization = VisualizationBase(params)
    visualization.visualize()