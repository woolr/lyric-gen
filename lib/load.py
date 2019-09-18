# Load

# Walk directories, get file names (see file copy project)
# Load .txt files
# get file names too
#

import os
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU


def load_input_texts():

    input_texts = []
    target_texts = []
    for line in open('Output.txt'):
        line = line.rstrip()
        if not line:
            continue

        input_line = '<SOS> ' + line
        target_line = line + ' <EOS>'

        input_texts.append(input_line)
        target_texts.append(target_line)

    all_lines = input_texts + target_texts

    return all_lines


def load_pretrained_wordvecs():
    # options for other word models?

    # load in pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}

    # TODO: add env var for path to models
    with open(os.path.join(f'glove.6B/glove.6B.{str(EMBEDDING_DIM)}sd.txt')) as f:
      # is just a space-separated text file in the format:
      # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    print('Found %s word vectors.' % len(word2vec))
