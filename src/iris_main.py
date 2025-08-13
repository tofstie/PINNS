import tensorflow as tf

from sklearn.datasets import load_iris

from src.pipeline.pipeline import Pipeline
from src.dataloader.dataloader_factory import DataLoaderFactory
from src.model.category_neural_network import CategoryNeuralNetwork

if __name__ == "__main__":
    iris = load_iris()
    X,Y = iris.data, iris.target

    input_shape = (X.shape[1],)
    model_architecture = [
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=input_shape),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ]

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    compile_parameters = {
        "optimizer": "adam",
        "loss": loss_function,
        "metrics": [tf.keras.metrics.SparseCategoricalAccuracy]
    }

    train_parameters = {
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.1,
        "scaler_required": False
    }

    dataloader = DataLoaderFactory().create_data_loader("category",test_size=0.2,random_state=42)
    NN = CategoryNeuralNetwork(model_layers=model_architecture)

    pipe = Pipeline(dataloader,NN,loss_function)
    pipe.run(
        features=X,
        labels=Y,
        compile_params = compile_parameters,
        train_param=train_parameters
    )