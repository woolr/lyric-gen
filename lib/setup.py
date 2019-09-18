# Setup


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
    max_sequence_length = min(max_sequence_length_from_data, user_max_sequence_length)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')
    print('Shape of data tensor:', input_sequences.shape)
    # TODO: Do I need to return max sequence length too?
    return input_sequences, target_sequences


def prep_embedding_matrix():
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


def generate_one_hot_targets():
    """

    """

    # one-hot the targets (can't use sparse cross-entropy)
    one_hot_targets = np.zeros((len(input_sequences), max_sequence_length, num_words))
    for i, target_sequence in enumerate(target_sequences):
        for t, word in enumerate(target_sequence):
            if word > 0:
                one_hot_targets[i, t, word] = 1

    return one_hot_targets

