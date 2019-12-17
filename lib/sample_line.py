# Sample Line
import numpy as np


def sample_line(sampling_model, latent_dimensions, word2idx, idx2word, max_sequence_length):
    # initial inputs
    np_input = np.array([[word2idx['<sos>']]])
    h = np.zeros((1, latent_dimensions))
    c = np.zeros((1, latent_dimensions))

    # so we know when to quit
    eos = word2idx['<eos>']

    # store the output here
    output_sentence = []

    for _ in range(max_sequence_length):
        o, h, c = sampling_model.predict([np_input, h, c])

        # print("o.shape:", o.shape, o[0,0,:10])
        # idx = np.argmax(o[0,0])
        probs = o[0, 0]
        if np.argmax(probs) == 0:
            print("wtf")
        probs[0] = 0
        probs /= probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        if idx == eos:
            break

        # accumulate output
        output_sentence.append(idx2word.get(idx, '<WTF %s>' % idx))

        # make the next input into model
        np_input[0, 0] = idx

    return ' '.join(output_sentence)
