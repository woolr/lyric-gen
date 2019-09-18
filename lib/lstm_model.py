# lstm_model

"""

Build Model

Compile model

"""

def make_embedding_layer():
    # load pre-trained word embeddings into an Embedding layer
    embedding_layer = Embedding(
      num_words,
      EMBEDDING_DIM,
      weights=[embedding_matrix],
      # trainable=False
    )


def build_model():
    """

    """
    # create an LSTM network with a single LSTM

    print('Building model...')
    input_ = Input(shape=(max_sequence_length,))
    initial_h = Input(shape=(latent_dimensions,))
    initial_c = Input(shape=(latent_dimensions,))
    x = embedding_layer(input_)
    lstm = LSTM(latent_dimensions, return_sequences=True, return_state=True)
    x, _, _ = lstm(x, initial_state=[initial_h, initial_c]) # don't need the states here
    dense = Dense(num_words, activation='softmax')
    output = dense(x)

    return [input_, initial_h, initial_c], output


def compile_model():
    """

    """
    model = Model([input_, initial_h, initial_c], output)
    model.compile(
      loss='categorical_crossentropy',
      # optimizer='rmsprop',
      optimizer=Adam(lr=0.01),
      # optimizer=SGD(lr=0.01, momentum=0.9),
      metrics=['accuracy'] # not meaningful here
    )

    return model
