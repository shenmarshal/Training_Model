"""
Model definition for CNN sentiment training

Author pharnoux

"""

import boto3
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Embedding


def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """

    print("Creating a model...")

    embeddings_matrix = load_embeddings(
        config["embeddings_path"], config["embeddings_dictionary_size"],
        config["embeddings_vector_size"])

    cnn_model = Sequential()
    embeddings = Embedding(
        config["embeddings_dictionary_size"], config["embeddings_vector_size"],
        weights=[np.array(embeddings_matrix)], input_length=config["padding_size"], trainable=True)
    cnn_model.add(embeddings)
    cnn_model.add(Conv1D(filters=100, kernel_size=2, padding="valid", activation="relu", strides=1))
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(256, activation="relu"))
    cnn_model.add(Dense(1, activation="sigmoid"))

    cnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return cnn_model

def load_embeddings(embeddings_path, embeddings_dictionary_size, embeddings_vector_size):
    """
    Loading GloVe embeddings

    """

    print("Loading embeddings at path {}...".format(embeddings_path))

    embeddings_matrix = np.zeros((embeddings_dictionary_size, embeddings_vector_size))

    s3_file = False

    if "s3://" in embeddings_path:
        s3_file = True
        s3_client = boto3.client("s3")

        path_split = embeddings_path.replace("s3://", "").split("/")

        bucket = path_split.pop(0)
        key = "/".join(path_split)

        data = s3_client.get_object(Bucket=bucket, Key=key)
        embeddings_file = data["Body"].iter_lines()

    else:
        embeddings_file = open(embeddings_path, "r")

    for index, line in enumerate(embeddings_file):

        if index >= embeddings_dictionary_size:
            break

        if s3_file:
            line = line.decode("utf8")

        split = line.split(" ")

        vector = np.asarray(split[1:], dtype="float32")

        embeddings_matrix[index] = vector

    return embeddings_matrix

def save_model(model, output):
    """
    Method to save a model

    """

    print("Saving model...")

    model.save(output, save_format="h5")

    print("Model successfully saved at: {}".format(output))
