# Setup
import numpy as np
from keras.layers import Embedding


def prep_embedding_matrix(word2idx, word2vec, max_vocab_size, embedding_dimension):
    """


    """
    print('Filling pre-trained embeddings...')
    num_words = min(max_vocab_size, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dimension))
    for word, i in word2idx.items():
        if i < max_vocab_size:
            embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, num_words


def generate_one_hot_targets(input_sequences, target_sequences, max_sequence_length, num_words):
    """

    """

    # one-hot the targets (can't use sparse cross-entropy)
    one_hot_targets = np.zeros((len(input_sequences), max_sequence_length, num_words))
    for i, target_sequence in enumerate(target_sequences):
        for t, word in enumerate(target_sequence):
            if word > 0:
                one_hot_targets[i, t, word] = 1

    return one_hot_targets


def make_embedding_layer(embedding_matrix, embedding_dimension, num_words):
    # load pre-trained word embeddings into an Embedding layer
    embedding_layer = Embedding(
        num_words,
        embedding_dimension,  # pass-thru
        weights=[embedding_matrix],
    )

    return embedding_layer
