"""
# generate

"""
import argparse
import matplotlib.pyplot as plt

from data_load import load_input_texts, load_pretrained_wordvecs
from process_text import tokenize_corpus, prepare_sequences
from embedding import prep_embedding_matrix, make_embedding_layer, generate_one_hot_targets
from lstm_model import build_model, compile_model, make_sample_model
from train import train_model
from sample_line import sample_line


def parse_args():
    parser = argparse.ArgumentParser(description="Args ")
    parser.add_argument('--file-path', type=str, default='~/data.txt')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--validation-split', type=float, default=0.2)
    parser.add_argument('--word-embedding-path', type=str, default='~/datasets/glove/')
    parser.add_argument('--word-embedding-dim', type=int, default=50)
    parser.add_argument('--max-vocab-size', type=int, default=20000)
    parser.add_argument('--max-sequence-length', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--prompt', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    latent_dim = args.latent_dim

    lines, input_texts, target_texts = load_input_texts(args.file_path)
    word2vec_map = load_pretrained_wordvecs(embedding_dimension=args.word_embedding_dim)

    input_seq, target_seq, tokenizer, word2idx, idx2word, max_sequence_length_from_data = \
        tokenize_corpus(lines, input_texts, target_texts, args.max_vocab_size)

    input_seq, target_seq, max_sequence_length = prepare_sequences(
        input_seq, target_seq, max_sequence_length_from_data, args.max_sequence_length)

    print(max_sequence_length)

    embedding_matrix, num_words = prep_embedding_matrix(
        word2idx, word2vec_map, args.max_vocab_size, args.word_embedding_dim)

    one_hot_targets = generate_one_hot_targets(
        input_seq, target_seq, max_sequence_length, num_words)

    embedding_layer = make_embedding_layer(embedding_matrix, args.word_embedding_dim, num_words)

    [input_, initial_h, initial_c], output = build_model(
        max_sequence_length, latent_dim, embedding_layer, num_words)

    model = compile_model(input_, initial_h, initial_c, output)

    trained_model = train_model(model, one_hot_targets, input_seq, latent_dim,
                                args.batch_size, args.epochs, args.validation_split)

    plt.plot(trained_model.history['loss'], label='loss')
    plt.plot(trained_model.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(trained_model.history['accuracy'], label='acc')
    plt.plot(trained_model.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

    sampling_model = make_sample_model(embedding_layer, latent_dim,
                                       initial_h, initial_c, num_words)

    # generate a 4 line verse
    while True:
        for _ in range(4):
            line = sample_line(sampling_model, latent_dim, word2idx,
                               idx2word, max_sequence_length, prompt=args.prompt)
            print(line)

        ans = input("More lyrics? [Y/n]---")
        if ans and ans[0].lower().startswith('n'):
            break
