"""
# generate

"""
import argparse
import matplotlib.pyplot as plt

from data_load import load_input_texts, load_pretrained_wordvecs
from process_text import tokenize_corpus, prepare_sequences
from embedding import prep_embedding_matrix, make_embedding_layer, generate_one_hot_targets
from lstm_model import build_model, compile_model
from train import train_model


# TODO: Make all these guys arguments
max_sequence_length = 100
max_vocab_size = 20000
embedding_dim = 50
validation_split = 0.2
batch_size = 128
epochs = 3
latent_dim = 25
verbose = True  # Set up proper ogging


def parse_args():
    parser = argparse.ArgumentParser(description="Args ")
    parser.add_argument('--file-path', type=str, default='~/data.txt')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    lines, input_texts, target_texts = load_input_texts(args.file_path)
    word2vec_map = load_pretrained_wordvecs(embedding_dimension=embedding_dim)

    input_sequences, target_sequences, tokenizer, word2idx, idx2word, max_sequence_length_from_data = \
        tokenize_corpus(lines, input_texts, target_texts, max_vocab_size)

    input_sequences, target_sequences, max_sequence_length = prepare_sequences(
        input_sequences, target_sequences, max_sequence_length_from_data, max_sequence_length)

    print(max_sequence_length)

    embedding_matrix, num_words = prep_embedding_matrix(
        word2idx, word2vec_map, max_vocab_size, embedding_dim)

    one_hot_targets = generate_one_hot_targets(
        input_sequences, target_sequences, max_sequence_length, num_words)

    embedding_layer = make_embedding_layer(embedding_matrix, embedding_dim, num_words)

    [input_, initial_h, initial_c], output = build_model(
        max_sequence_length, latent_dim, embedding_layer, num_words)

    model = compile_model(input_, initial_h, initial_c, output)

    r = train_model(model, one_hot_targets, input_sequences, latent_dim,
                    batch_size, epochs, validation_split)

    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()
