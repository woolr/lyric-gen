# Train models

"""

Train

Save trained model

"""

def train_model():
    # Initialize h and c with zeroes
    z = np.zeros((len(input_sequences), latent_dimensions))

    # Fit the model to the input_sequences
    r = model.fit(
      [input_sequences, z, z],
      one_hot_targets,
      batch_size=batch_size,
      epochs=num_epochs,
      validation_split=validation_split
    )
