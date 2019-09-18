# Process
import os
import sys
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import keras.backend as K

# If there are GPUs availiable to use - import the GPU specific layers
if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU


def tokenize_corpus():
    """
    Tokenize a corpus using Keras preprocessing Tokenizer (LINK)

    Parameters
    ----------

    Returns
    ----------

    Raises
    ----------
    """

    # convert the sentences (strings) into integers
    # filters='' - means that we don't filter out special characters like "<" and ">"
    tokenizer = Tokenizer(num_words=max_vocab_size, filters='')
    tokenizer.fit_on_texts(all_lines)
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)

    # find max sequence length
    max_sequence_length_from_data = max(len(s) for s in input_sequences)
    if verbose:
        print('Max sequence length:', max_sequence_length_from_data)

    # get word 2 integer mapping
    word2idx = tokenizer.word_index
    if verbose:
        print('Found %s unique tokens.' % len(word2idx))
    assert('<SOS>' in word2idx)
    assert('<EOS>' in word2idx)

    return input_sequences, target_sequences, tokenizer, word2idx


def pad_sequences():
    """
    Ensures that sequences all have the same length

    TODO:  Explain reason for this

    Parameters
    ----------

    Returns
    ----------

    Raises
    ----------
    """
    # pad sequences so that we get a N x T matrix
    max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')
    print('Shape of data tensor:', input_sequences.shape)
    # TODO: Do I need to return max sequence length too?
    return input_sequences, target_sequences
