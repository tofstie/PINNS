import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.io import loadmat

from src.pipeline.pipeline import Pipeline
from src.dataloader.dataloader_factory import DataLoaderFactory
from src.model.advection_neural_network import AdvectionNeuralNetwork
from src.loss.advection_loss import AdvectionLoss

from src.visualization.visualization_parameters import VisualizationParameters, FigureParameters, DataParameters
from src.visualization.visualization_base import VisualizationBase

if __name__ == "__main__":

    advection_parameters = {
        "residual_factor": 0.1,
        "initial_condition_factor": 0.1,
        "initial_position": "cos",
        "boundary_condition": "periodic",
        "wave_speed": tf.constant(2.0,dtype=tf.float64),
        "dimension": 2
    }

    train_parameters = {
        "epochs": 20,
        "batch_size": 128,
        "validation_split": 0.05,
        "scaler_required": True
    }

    data_path = os.path.join("..", "data", "ADVECTION2D.mat")
    advection_data = loadmat(data_path)
    u = advection_data["u_solutions"]
    t = advection_data["T"].flatten()[:, None]
    x = advection_data["x"].flatten()[:, None]
    y = advection_data["y"].flatten()[:, None]
    X,Y,T = np.meshgrid(x,y,t,indexing='xy')

    INPUT = np.hstack((T.flatten()[:,None], X.flatten()[:,None], Y.flatten()[:,None]))
    OUTPUT = u.flatten()[:,None]

    input_shape = (INPUT.shape[1],)
    model_architecture = [
        tf.keras.layers.Dense(1+advection_parameters["dimension"], activation=tf.nn.tanh, input_shape=input_shape, dtype=tf.float64),
        tf.keras.layers.Dense(50, activation=tf.nn.tanh, dtype=tf.float64),
        tf.keras.layers.Dense(50, activation=tf.nn.tanh, dtype=tf.float64),
        tf.keras.layers.Dense(50, activation=tf.nn.tanh, dtype=tf.float64),
        tf.keras.layers.Dense(50, activation=tf.nn.tanh, dtype=tf.float64),
        tf.keras.layers.Dense(50, activation=tf.nn.tanh, dtype=tf.float64),
        tf.keras.layers.Dense(50, activation=tf.nn.tanh, dtype=tf.float64),
        tf.keras.layers.Dense(1, dtype=tf.float64)
    ]



    loss_function = AdvectionLoss(advection_parameters)
    learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=train_parameters['epochs'],
        decay_rate=0.98,
        staircase=True
    )
    optimizer_adam = tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler)
    compile_parameters = {
        "optimizer": optimizer_adam,
        "loss": loss_function,
        "metrics": [tf.keras.metrics.MeanAbsoluteError()]
    }

    ## Dataloader
    dataloader = DataLoaderFactory.create_data_loader("advection", test_size=0.2, random_state=42)

    ## Build NN
    NN = AdvectionNeuralNetwork(model_architecture, advection_parameters=advection_parameters)
    NN.set_loss_function(loss_function)
    loss_function.model = NN

    ## Create pipeline
    pipe = Pipeline(dataloader, NN, loss_function)

    pipe.run(
        features=INPUT,
        labels=OUTPUT,
        compile_params=compile_parameters,
        train_param=train_parameters
    )

    fig, ax = plt.subplots(1,1)
    ax.plot(NN.iter_loss[:,0],NN.iter_loss[:,1],label = "Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    plt.show()

    sample_idx = 61
    time_sample = t[sample_idx]
    X_samp, Y_samp, T_samp = np.meshgrid(x, y, time_sample, indexing='xy')
    inputs_for_plot = np.hstack((X_samp.flatten()[:,None], Y_samp.flatten()[:,None], T_samp.flatten()[:,None]))
    inputs = dataloader.scaler.transform(inputs_for_plot)
    output = NN.model(inputs, training = False)

    fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
    ax.plot_surface(np.squeeze(X_samp),np.squeeze(Y_samp), np.squeeze(np.reshape(output.numpy(),[50,50,1])))
    plt.show()
    fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
    ax.plot_surface(np.squeeze(X_samp),np.squeeze(Y_samp), np.squeeze(u[:,:,sample_idx]))
    plt.show()
    # ## Setting up Output Figure
    # params = VisualizationParameters()
    # output_fig_params = FigureParameters()
    # output_fig_params.dims = 3
    # output_fig_params.type = "subplot"
    # output_fig_params.subplot_rows = 3
    # output_fig_params.subplot_cols = 1
    # output_fig_params.add_x_axes_settings("Time (s)")
    # output_fig_params.add_y_axes_settings(r'theta', yscale = "linear")
    # output_fig_params.add_y_axes_settings(r'theta Sample', yscale="linear")
    # output_fig_params.add_y_axes_settings(r'L1 error', yscale = "linear")
    # output_fig_params.use_legend = False
    # ## Setting up data
    # PINN_data = DataParameters(x = X_samp.flatten()[:,None], y = Y_samp.flatten()[:,None], z = output,label = "PINN",plot_type="surf")
    # Actual_data = DataParameters(x = X_samp.flatten()[:,None], y = Y_samp.flatten()[:,None],z = u[:,:,sample_idx].flatten()[:,None],label = "Original Model",plot_type="surf")
    # Error_data = DataParameters(x = X_samp.flatten()[:,None], y = Y_samp.flatten()[:,None] ,z = np.abs(output - u[:,:,sample_idx].flatten()[:,None]),label = "L2 Error",plot_type="surf")
    #
    # output_fig_params.add_data(PINN_data,0)
    # output_fig_params.add_data(Actual_data,1)
    # output_fig_params.add_data(Error_data,2)
    #
    # params.figures.append(output_fig_params)
    # ## Visualize data
    # visualization = VisualizationBase(params)
    # visualization.visualize()