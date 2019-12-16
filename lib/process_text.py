# Process

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# # If there are GPUs availiable to use - import the GPU specific layers
# if len(K.tensorflow_backend._get_available_gpus()) > 0:
#     from keras.layers import CuDNNLSTM as LSTM
#     from keras.layers import CuDNNGRU as GRU


def tokenize_corpus(all_lines, input_texts, target_texts, max_vocab_size=20000, verbose=True):
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
    tokenizer = Tokenizer(num_words=max_vocab_size, filters='', lower=False)
    tokenizer.fit_on_texts(all_lines)
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)

    # find max sequence length from the actual data
    max_sequence_length_from_data = max(len(s) for s in input_sequences)
    if verbose:
        print('Max sequence length:', max_sequence_length_from_data)

    # get word 2 integer mapping
    word2idx = tokenizer.word_index
    if verbose:
        print('Found %s unique tokens.' % len(word2idx))
    assert('<SOS>' in word2idx)
    assert('<EOS>' in word2idx)

    # Get words from index
    idx2word = {v: k for k, v in word2idx.items()}

    return (input_sequences, target_sequences, tokenizer,
            word2idx, idx2word, max_sequence_length_from_data)


def prepare_sequences(input_sequences, target_sequences, msl_data, msl_user):
    """
    Ensures that sequences all have the same length

    Parameters
    ----------

    Returns
    ----------

    Raises
    ----------
    """
    # pad sequences so that we get a N x T matrix
    max_sequence_length = min(msl_data, msl_user)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')
    print('Shape of data tensor:', input_sequences.shape)

    return input_sequences, target_sequences, max_sequence_length
