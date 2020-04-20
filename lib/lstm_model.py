# lstm_model

"""

Build Model

Compile model

"""
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_model(max_sequence_length, latent_dimensions, embedding_layer, num_words):
    """
        Create an LSTM network with a single LSTM
    """
    print('Building model...')
    input_ = Input(shape=(max_sequence_length,))
    initial_h = Input(shape=(latent_dimensions,))
    initial_c = Input(shape=(latent_dimensions,))
    x = embedding_layer(input_)
    lstm = LSTM(latent_dimensions, return_sequences=True, return_state=True)
    x, _, _ = lstm(x, initial_state=[initial_h, initial_c])  # don't need the states here
    dense = Dense(num_words, activation='relu')
    dense = Dense(num_words, activation='softmax')
    output = dense(x)

    return [input_, initial_h, initial_c], output


def compile_model(input_, initial_h, initial_c, output):
    """
        Compile the keras model
    """
    model = Model([input_, initial_h, initial_c], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']  # not meaningful here
    )

    return model


def make_sample_model(embedding_layer, latent_dimensions,
                      initial_h, initial_c, num_words, save=False):
    """
        Sample using the trained model weights
    """
    # make a sampling model
    input_layer = Input(shape=(1,))  # only input one word at a time
    x = embedding_layer(input_layer)

    # now we need states to feed back in
    lstm = LSTM(latent_dimensions, return_sequences=True, return_state=True)

    x, h, c = lstm(x, initial_state=[initial_h, initial_c])
    dense = Dense(num_words, activation='softmax')
    output_layer = dense(x)

    sampling_model = Model([input_layer, initial_h, initial_c], [output_layer, h, c])
    if save:
        sampling_model.save("~/model_name.lstmmodel")
    return sampling_model
