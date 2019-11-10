import nltk
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from IPython.display import SVG
from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing
import mlflow
import time

DATA_DIRECTORY = './petfinder_dataset/'
SW = set(stopwords.words("english"))
MAX_SEQUENCE_LEN = 70
TARGET_COL = 'AdoptionSpeed'
BATCH_SIZE = 128
EMBEDDINGS_DIM = 100  # Given by the model (in this case glove.6B.100d)
HIDDEN_LAYER_SIZE = 128
EPOCHS = 20
FILTER_WIDTHS = [2, 3, 5]  # Take 2, 3, and 5 words
FILTER_COUNT = 64


def get_train_size(dataset):
    return int(dataset.shape[0] * 0.8)

def get_dev_size(dataset):
    return dataset.shape[0] - get_train_size(dataset)

def get_vocabulary(dataset):
    vocabulary = corpora.Dictionary(dataset["TokenizedDescription"])
    vocabulary.filter_extremes(no_below=1, no_above=1.0, keep_n=10000)
    return vocabulary
    
def get_nlabels(dataset):
    return dataset[TARGET_COL].unique().shape[0]

def tokenize_description(description):
    return [w.lower() for w in word_tokenize(description, language="english") if w.lower() not in SW]

def dataset_generator(ds, one_hot_columns, embedded_columns, numeric_columns, test_data=False):
    vocabulary = get_vocabulary(ds)

    for _, row in ds.iterrows():
        instance = {}
        
        # One hot encoded features
        instance["direct_features"] = np.hstack([
            tf.keras.utils.to_categorical(row[one_hot_col] - 1, max_value)
            for one_hot_col, max_value in one_hot_columns.items()
        ])

        # Numeric features (should be normalized beforehand)
        for num_column in numeric_columns:
            instance[num_column] = [row[num_column]]
        
        # Embedded features
        for embedded_col in embedded_columns:
            instance[embedded_col] = [row[embedded_col]]
        
        # Document to indices for text data, truncated at MAX_SEQUENCE_LEN words
        instance["description"] = vocabulary.doc2idx(
            row["TokenizedDescription"],
            unknown_word_index=len(vocabulary)
        )[:MAX_SEQUENCE_LEN]
        
        # One hot encoded target for categorical crossentropy
        if not test_data:
            nlabels = get_nlabels(ds)
            target = tf.keras.utils.to_categorical(row[TARGET_COL], nlabels)
            yield instance, target
        else:
            yield instance


def main():
    #nltk.download(["punkt", "stopwords"]);
    dataset = pd.read_csv(os.path.join(DATA_DIRECTORY, 'train.csv'))    
    # Fill the null values with the empty string to avoid errors with NLTK tokenization
    dataset["TokenizedDescription"] = dataset["Description"].fillna(value="").apply(tokenize_description)
    nlabels = get_nlabels(dataset)

    vocabulary = get_vocabulary(dataset)
    embeddings_index = {}

    with open(os.path.join(DATA_DIRECTORY, 'glove.6B.100d.txt'), "r") as fh:
        for line in fh:
            values = line.split()
            word = values[0]
            if word in vocabulary.token2id:  # Only use the embeddings of words in our vocabulary
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

    print("Found {} word vectors.".format(len(embeddings_index)))

    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: dataset[one_hot_col].max()
        for one_hot_col in ['Gender', 'Color1', 'Color2', 'Color3',
                            'Type', 'MaturitySize', 'FurLength', 'State',
                            'Vaccinated', 'Dewormed', 'Sterilized', 'Health']
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in ['Breed1', 'Breed2']
    }

    numeric_columns = ['Age', 'Fee', 'Quantity']
    for num_column in numeric_columns:
        dataset[num_column] = preprocessing.StandardScaler().fit_transform(dataset[[num_column]].values)
    
    # Set output types of the generator (for numeric types check the type is valid)
    instance_types = {
        "direct_features": tf.float32,
        "description": tf.int32
    }

    for embedded_col in embedded_columns:
        instance_types[embedded_col] = tf.int32
    
    for numeric_col in numeric_columns:
        instance_types[numeric_col] = tf.int32
            
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(dataset, one_hot_columns, embedded_columns, numeric_columns),
        output_types=(instance_types, tf.int32)
    )

    shuffled_dataset = tf_dataset.shuffle(get_train_size(dataset) + get_dev_size(dataset), seed=42)

    # Pad the datasets to the max value for all the "non sequence" features
    padding_shapes = (
        {k: [-1] for k in ["direct_features"] + list(embedded_columns.keys()) + numeric_columns},
        [-1]
    )

    # Pad to MAX_SEQUENCE_LEN for sequence features
    padding_shapes[0]["description"] = [MAX_SEQUENCE_LEN]

    # Pad values are irrelevant for non padded data
    padding_values = (
        {k: 0 for k in list(embedded_columns.keys()) + numeric_columns},
        0
    )

    # Padding value for direct features should be a float
    padding_values[0]["direct_features"] = np.float32(0)

    # Padding value for sequential features is the vocabulary length + 1
    padding_values[0]["description"] = len(vocabulary) + 1

    train_dataset = shuffled_dataset.skip(get_dev_size(dataset))\
        .padded_batch(BATCH_SIZE, padded_shapes=padding_shapes, padding_values=padding_values)

    dev_dataset = shuffled_dataset.take(get_dev_size(dataset))\
        .padded_batch(BATCH_SIZE, padded_shapes=padding_shapes, padding_values=padding_values)
    
    embedding_matrix = np.zeros((len(get_vocabulary(dataset)) + 2, 100))

    for widx, word in get_vocabulary(dataset).items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[widx] = embedding_vector
        else:
            # Random normal initialization for words without embeddings
            embedding_matrix[widx] = np.random.normal(size=(100,))  

    # Random normal initialization for unknown words
    embedding_matrix[len(get_vocabulary(dataset))] = np.random.normal(size=(100,))

    tf.keras.backend.clear_session()

    # Add one input and one embedding for each embedded column
    embedding_layers = []
    inputs = []
    for embedded_col, max_value in embedded_columns.items():
        input_layer = tf.keras.layers.Input(shape=(1,), name=embedded_col)
        inputs.append(input_layer)
        # Define the embedding layer
        embedding_size = int(max_value / 4)
        embedding_layers.append(
            tf.squeeze(
                tf.keras.layers.Embedding(
                    input_dim=max_value, 
                    output_dim=embedding_size
                )(input_layer), 
                axis=-2
            )
        )
        print('Adding embedding of size {} for layer {}'.format(embedding_size, embedded_col))

    # Add the direct features already calculated
    direct_features_input = tf.keras.layers.Input(
        shape=(sum(one_hot_columns.values()),), 
        name='direct_features'
    )
    inputs.append(direct_features_input)

    # Word embedding layer
    description_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LEN,), name="description")
    inputs.append(description_input)

    word_embeddings_layer = tf.keras.layers.Embedding(
        embedding_matrix.shape[0],
        EMBEDDINGS_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LEN,
        trainable=False,
        name="word_embedding"
    )
    
    # Create a NN (CNN or RNN) for the description input (replace the next)
    embedded_description = word_embeddings_layer(description_input)
    conv_layers = []
    for filter_width in FILTER_WIDTHS:
        layer = tf.keras.layers.Conv1D(
            FILTER_COUNT,
            filter_width,
            activation="relu",
            name="conv_{}_words".format(filter_width)
        )(embedded_description)
        layer = tf.keras.layers.GlobalMaxPooling1D(name="max_pool_{}_words".format(filter_width))(layer)
        conv_layers.append(layer)

    convolved_features = tf.keras.layers.Concatenate(name="convolved_features")(conv_layers)
    hidden_layer = tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation="relu")(convolved_features)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(hidden_layer)

    model = tf.keras.models.Model(inputs=inputs, outputs=[output_layer], name="crap_model")
    
    model.compile(loss='categorical_crossentropy', 
              optimizer='nadam',
              metrics=['accuracy'])
    model.summary()
    
    mlflow.set_experiment('newbie_approach')

    with mlflow.start_run(nested=True):
        mlflow.log_param('hidden_layer_size', HIDDEN_LAYER_SIZE)
        mlflow.log_param('embedded_columns', embedded_columns)
        mlflow.log_param('one_hot_columns', one_hot_columns)
        mlflow.log_param('numberic_columns', numeric_columns)
        
        # Train
        history = model.fit(train_dataset, epochs=EPOCHS)
        
        # Evaluate
        loss, accuracy = model.evaluate(dev_dataset, verbose=0)
        print("\n*** Validation loss: {} - accuracy: {}".format(loss, accuracy))
        mlflow.log_metric('epochs', EPOCHS)
        mlflow.log_metric('train_loss', history.history["loss"][-1])
        mlflow.log_metric('train_accuracy', history.history["accuracy"][-1])
        mlflow.log_metric('validation_loss', loss)
        mlflow.log_metric('validation_accuracy', accuracy)
    
    test_dataset = pd.read_csv(os.path.join(DATA_DIRECTORY, 'test.csv'))

    # First tokenize the description
    test_dataset["TokenizedDescription"] = test_dataset["Description"]\
        .fillna(value="").apply(tokenize_description)

    # Generate the basic TF dataset
    tf_test_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(test_dataset, one_hot_columns, embedded_columns, numeric_columns, True),
        output_types=instance_types  # It should have the same instance types
    )

    test_data = tf_test_dataset.padded_batch(BATCH_SIZE, 
        padded_shapes=padding_shapes[0], 
        padding_values=padding_values[0]
    )
    
    test_dataset["AdoptionSpeed"] = model.predict(test_data).argmax(axis=1)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = "./submission_" + timestr + ".csv"
    test_dataset.to_csv(filename, index=False, columns=["PID", "AdoptionSpeed"])

    mlflow.log_param('filename', filename)
    print ("That's all for now")

if __name__ == '__main__':
    main()
