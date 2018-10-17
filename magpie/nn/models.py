from keras.layers.convolutional import MaxPooling1D, Convolution1D
from keras.layers import GlobalAveragePooling1D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers import Merge
from keras.layers.normalization import BatchNormalization
from keras.layers import Embedding
from keras.layers.recurrent import GRU
from keras.models import Sequential

from magpie.config import SAMPLE_LENGTH
from magpie.config import EMBEDDING_SIZE

def get_nn_model(nn_model, embedding, output_length):
    if nn_model == 'cnn':
        return cnn(input_shape=embedding, output_length=output_length)
    elif nn_model == 'rnn':
        return rnn(input_shape=embedding, output_length=output_length)
    elif nn_model == 'fastText': 
        return fastText(input_shape=embedding, output_length=output_length)
    else:
        raise ValueError("Unknown NN type: {}".format(nn_model))


def cnn(input_shape, output_length):
    """ Create and return a keras model of a CNN """
    NB_FILTER = 256
    NGRAM_LENGTHS = [1, 2, 3, 4, 5]

    conv_layers = []
    for ngram_length in NGRAM_LENGTHS:
        ngram_layer = Sequential()
        ngram_layer.add(Convolution1D(
            NB_FILTER,
            ngram_length,
            input_dim=input_shape,
            input_length=SAMPLE_LENGTH,
            init='lecun_uniform',
            activation='tanh',
        ))
        pool_length = SAMPLE_LENGTH - ngram_length + 1
        ngram_layer.add(MaxPooling1D(pool_length=pool_length))
        conv_layers.append(ngram_layer)

    model = Sequential()
    model.add(Merge(conv_layers, mode='concat'))

    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(output_length, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'],
    )

    return model


def rnn(input_shape, output_length):
    """ Create and return a keras model of a RNN """
    HIDDEN_LAYER_SIZE = 256

    model = Sequential()

    model.add(GRU(
        HIDDEN_LAYER_SIZE,
        input_dim=input_shape,
        input_length=SAMPLE_LENGTH,
        init='glorot_uniform',
        inner_init='normal',
        activation='relu',
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(output_length, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'],
    )

    return model


def fastText(input_shape, output_length): 
    
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(input_shape,
        EMBEDDING_SIZE,
        input_length=SAMPLE_LENGTH))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(GlobalAveragePooling1D())

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(output_length, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])

    return model
