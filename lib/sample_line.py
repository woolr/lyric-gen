# Sample Line
import numpy as np


def sample_line(sampling_model, latent_dimensions, word2idx, idx2word, max_sequence_length,
                prompt=None):
    # initial inputs
    np_input = np.array([[word2idx['<sos>']]])
    hidden = np.zeros((1, latent_dimensions))
    cell = np.zeros((1, latent_dimensions))

    # so we know when to quit
    eos = word2idx['<eos>']

    # store the output here
    output_sentence = []

    for number in range(max_sequence_length):
        out, hidden, cell = sampling_model.predict([np_input, hidden, cell])
        if prompt:
            prompt_words = prompt.split("-")
            assert len(prompt_words) < 4

            for i, word in enumerate(prompt_words):
                word_i = word2idx[word]
                output_sentence.append(idx2word.get(word_i, '<WTF %s>' % word_i))
                if i == len(prompt_words):
                    break
            prompt = None
            # print("out.shape:", out.shape, out[0, 0, :10])
            idx = np.argmax(out[0, 0])

        probs = out[0, 0]
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
