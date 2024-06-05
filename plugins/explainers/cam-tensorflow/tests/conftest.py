import pytest
import numpy as np


@pytest.fixture(scope="session")
def custom_tf_CNN():
    """
    Creates and trains a simple TF CNN on the mnist dataset.
    Returns the model, a subset of the test dataset and the class names.

    Taken from https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/deep_explainer/Front%20Page%20DeepExplainer%20MNIST%20Example.html
    """
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    from tensorflow.keras import backend as K

    batch_size = 128
    num_classes = 10
    epochs = 3

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_first":
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype("float32")[:500]
    X_test = X_test.astype("float32")[:100]
    X_train /= 255
    X_test /= 255
    y_train = y_train[:500]
    y_test = y_test[:100]
    print("x_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(), metrics=["accuracy"]
    )

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

    return model, X_test[:10], [str(i) for i in range(10)], y_test[:10]


@pytest.fixture(scope="session")
def dog_cat_image():
    """Loads the cat-dog image exampe from imagenet."""
    from PIL import Image
    import requests
    from io import BytesIO

    response = requests.get("https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png")
    return np.array(Image.open(BytesIO(response.content)))


@pytest.fixture(scope="session")
def tf_VGG():
    """Loads the keras.applications VGG16 pretrained on imagenet"""
    from tensorflow.keras.applications import VGG16

    return VGG16(weights="imagenet")


@pytest.fixture(scope="session")
def tf_resnet50():
    """Loads the keras.applications ResNet50 pretrained on imagenet"""
    from tensorflow.keras.applications.resnet50 import ResNet50

    return ResNet50(weights="imagenet")
