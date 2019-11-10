import argparse

import os
import mlflow
import numpy
import pandas
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn import preprocessing
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize

TARGET_COL = 'AdoptionSpeed'


def read_args():
    parser = argparse.ArgumentParser(
        description='Training a MLP on the petfinder dataset')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--dataset_dir', default='../petfinder_dataset', type=str,
                        help='Directory with the training and test files.')
    parser.add_argument('--experiment_name', type=str, default='Base model',
                        help='Name of the experiment, used in mlflow.')
    args = parser.parse_args()
    return args


def process_features(df, one_hot_columns, numeric_columns, embedded_columns, test=False):
    direct_features = []

    # Create one hot encodings (every column here is int64, so no need to
    # specify 'dtype' to 'to_categorical'
    for one_hot_col, max_value in one_hot_columns.items():
        direct_features.append(tf.keras.utils.to_categorical(df[one_hot_col] - 1, max_value))

    for num_column in numeric_columns:
        scaled_features = preprocessing.StandardScaler().fit_transform(df[[num_column]].values)
        # scaled_features = preprocessing.MinMaxScaler().fit_transform(df[[num_column]].values)
        direct_features.append(scaled_features)

    # Concatenate all features that don't need further embedding into a single matrix.
    features = {'direct_features': numpy.hstack(direct_features)}

    # Create embedding columns - nothing to do here. We will use the zero embedding for OOV
    for embedded_col in embedded_columns.keys():
        features[embedded_col] = df[embedded_col].values

    if not test:
        nlabels = df[TARGET_COL].unique().shape[0]
        # Convert labels to one-hot encodings
        targets = tf.keras.utils.to_categorical(df[TARGET_COL], nlabels)
    else:
        targets = None

    return features, targets


def load_dataset(dataset_dir, test_size):
    # Read train dataset (and maybe dev, if you need to...)
    dataset, dev_dataset = train_test_split(
        pandas.read_csv(os.path.join(dataset_dir, 'train.csv')), test_size=test_size)

    test_dataset = pandas.read_csv(os.path.join(dataset_dir, 'test.csv'))

    print('Training samples {}, dev_samples {}, test_samples {}'.format(
        dataset.shape[0], dev_dataset.shape[0], test_dataset.shape[0]))

    return dataset, dev_dataset, test_dataset


def get_search_space():
    search_space = {
        "epochs": Integer(1, 70, name="epochs"),
        "batch_size": Integer(8, 64, name="batch_size"),
        "test_size": Real(low=0.1, high=0.3, name='test_size'),
        "hidden_layer_1": Integer(5, 100, name="hidden_layer_1"),
        "hidden_layer_2": Integer(5, 100, name="hidden_layer_2"),
        "hidden_layer_3": Integer(5, 100, name="hidden_layer_3"),
        "dropout_1": Real(low=0.0, high=0.4, name="dropout_1"),
        "dropout_2": Real(low=0.0, high=0.4, name="dropout_2"),
        "dropout_3": Real(low=0.0, high=0.4, name="dropout_3"),
        "learning_rate": Real(low=1e-4, high=1e-1, prior='log-uniform', name='learning_rate')
    }
    return search_space

def hyperparam_value(param_name, param_list):
    search_space = get_search_space()
    search_space_keys, search_space_vals = zip(*search_space.items())
    search_space_keys = {param_name: idx
                         for idx, param_name in enumerate(search_space_keys)}
    return param_list[search_space_keys[param_name]]

def print_selected_hyperparams(param_values):
    search_space = get_search_space()
    search_space_keys, search_space_vals = zip(*search_space.items())
    search_space_keys = {param_name: idx
                         for idx, param_name in enumerate(search_space_keys)}
    for param_name in search_space_keys:
        print("\t", param_name, hyperparam_value(param_name, param_values))

def show_best(res):
    search_space = get_search_space()
    search_space_keys, search_space_vals = zip(*search_space.items())
    search_space_keys = {param_name: idx
                         for idx, param_name in enumerate(search_space_keys)}
    print("Best value: %.4f" % res.fun)
    param_names = {idx: param_name for param_name, idx in search_space_keys.items()}
    best_params = {param_names[i]: param_value for i, param_value in enumerate(res.x)}
    print("Best params:")
    print(best_params)

def objective_function(params):
    print_selected_hyperparams(params)

    test_size = hyperparam_value("test_size", params)
    batch_size = hyperparam_value("batch_size", params)
    epochs = hyperparam_value("epochs", params)
    learning_rate = hyperparam_value("learning_rate", params)
    hidden_layer_1 = hyperparam_value("hidden_layer_1", params)
    hidden_layer_2 = hyperparam_value("hidden_layer_2", params)
    hidden_layer_3 = hyperparam_value("hidden_layer_3", params)
    dropout_1 = hyperparam_value("dropout_1", params)
    dropout_2 = hyperparam_value("dropout_2", params)
    dropout_3 = hyperparam_value("dropout_3", params)

    args = read_args()
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, test_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]

    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: max(dataset[one_hot_col].max(), dev_dataset[one_hot_col].max(), test_dataset[one_hot_col].max())
        for one_hot_col in ['Gender', 'Color1', 'Color2', 'Color3',
                            'Type', 'MaturitySize', 'FurLength', 'State',
                            'Vaccinated', 'Dewormed', 'Sterilized', 'Health']
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in ['Breed1', 'Breed2']
    }
    numeric_columns = ['Age', 'Fee', 'Quantity']

    # TODO (optional) put these three types of columns in the same dictionary with "column types"
    X_train, y_train = process_features(dataset, one_hot_columns, numeric_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numeric_columns, embedded_columns)

    # TODO shuffle the train dataset!
    #    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    dev_ds = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(process_features(
        test_dataset, one_hot_columns, numeric_columns, embedded_columns, test=True)[0]).batch(batch_size)

    test_pids = test_dataset["PID"]

    # Build the Keras model

    tf.keras.backend.clear_session()

    # Add one input and one embedding for each embedded column
    embedding_layers = []
    inputs = []
    for embedded_col, max_value in embedded_columns.items():
        input_layer = layers.Input(shape=(1,), name=embedded_col)
        inputs.append(input_layer)
        # Define the embedding layer
        embedding_size = int(max_value / 4)
        embedding_layers.append(
            tf.squeeze(layers.Embedding(input_dim=max_value, output_dim=embedding_size)(input_layer), axis=-2))
        print('Adding embedding of size {} for layer {}'.format(embedding_size, embedded_col))

    # Add the direct features already calculated
    direct_features_input = layers.Input(shape=direct_features_input_shape, name='direct_features')
    inputs.append(direct_features_input)

    # Concatenate everything together
    features = layers.concatenate(embedding_layers + [direct_features_input])

    # Generate Dense layers with hidden_layer_sizes, and then
    # Dropout layers with dropout ratios
    ls = zip([hidden_layer_1, hidden_layer_2, hidden_layer_3], [dropout_1, dropout_2, dropout_3])
    dense = []
    dropout = []
    for order, (layer_size, dropout_ratio) in enumerate(ls):
        if order == 0:
            dense.append(layers.Dense(layer_size, activation='relu')(features))
        else:
            dense.append(layers.Dense(layer_size, activation='relu')(dropout[order - 1]))
        dropout.append(layers.Dropout(dropout_ratio)(dense[order]))

    # output_layer = layers.Dense(nlabels, activation='softmax')(dropout[len(args.dropout) - 1])
    output_layer = layers.Dense(nlabels, activation='softmax')(dropout[2])

    # Build model
    model = models.Model(inputs=inputs, outputs=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    model.summary()

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(nested=True):
        # Log model hiperparameters first
        mlflow.log_param('hidden_layer_1', hidden_layer_1)
        mlflow.log_param('hidden_layer_2', hidden_layer_2)
        mlflow.log_param('hidden_layer_3', hidden_layer_3)
        mlflow.log_param('dropout_1', dropout_1)
        mlflow.log_param('dropout_2', dropout_2)
        mlflow.log_param('dropout_3', dropout_3)
        mlflow.log_param('embedded_columns', embedded_columns)
        mlflow.log_param('one_hot_columns', one_hot_columns)
        mlflow.log_param('test_size', test_size)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('numeric_columns', numeric_columns)
        mlflow.log_param('epochs', epochs)
        # Train
        # history = model.fit(train_ds, epochs=epochs)
        history = model.fit(train_ds, epochs=epochs)

        # TODO: analyze history to see if model converges/overfits

        # loss, accuracy = 0, 0
        loss, accuracy = model.evaluate(dev_ds)

        print("*** Dev loss: {} - accuracy: {}".format(loss, accuracy))
        mlflow.log_metric('loss', loss)
        mlflow.log_metric('accuracy', accuracy)

        predictions = model.predict(test_ds)
 
        # Convert predictions to classes
        labels = numpy.argmax(predictions, axis=-1)

        # Save the results for submission
        timestr = time.strftime("%Y%m%d-%H%M%S")
        submission = pandas.DataFrame(list(zip(test_pids, labels)), columns=["PID", "AdoptionSpeed"])
        filename = "./submission_" + timestr + ".csv"
        submission.to_csv(filename, header=True, index=False)
        mlflow.log_param('filename', filename)
    return (accuracy * (-1))


def main():
    search_space = get_search_space()
    search_space_keys, search_space_vals = zip(*search_space.items())
    search_space_keys = {param_name: idx
                         for idx, param_name in enumerate(search_space_keys)}
    iterations = 20
    exploration_result = gp_minimize(objective_function, search_space_vals, random_state=21, verbose=1,
                                     n_calls=iterations)
    show_best(exploration_result)

    print('All operations completed')


if __name__ == '__main__':
    main()

