# Train models

"""

Train

Save trained model

"""
import numpy as np


def train_model(model, one_hot_targets, input_sequences, latent_dimensions,
                batch_size, num_epochs, validation_split):
    """

    """
    # Initialize h and c with zeroes
    z = np.zeros((len(input_sequences), latent_dimensions))

    # Fit the model to the input_sequences
    trained_model = model.fit(
        [input_sequences, z, z],
        one_hot_targets,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=validation_split
    )

    return trained_model
