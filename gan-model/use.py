import tensorflow as tf
import numpy as np

# Load pre-trained models
generator = tf.keras.models.load_model('generator_model.keras')
discriminator = tf.keras.models.load_model('discriminator_model.keras')

# Create a combined model
gan_model = tf.keras.Sequential()
gan_model.add(generator)
gan_model.add(discriminator)

# Compile the combined model
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

# Define the length of the initial sequence and the number of characters to be generated
seq_length = 50
num_chars = 20

# Define initial sequence
seed_text = "Hello"
# Load data and define char_to_int, vocab_size, int_to_char
data = open('C:/Users/romai/OneDrive/Bureau/ai prise/bot chat/data.txt', 'r').read()
chars = list(set(data))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
seq_length = 50

# Generate text with the GAN template
generated_text = seed_text
for _ in range(num_chars):
# Adjust initial sequence length
    current_seq = ' ' * (seq_length - len(seed_text)) + seed_text

  # Encode sequence
    encoded_seq = np.array([char_to_int[char] for char in current_seq])
    encoded_seq = np.eye(vocab_size)[encoded_seq.reshape(-1)]
    encoded_seq = encoded_seq.reshape(1, seq_length, vocab_size)

  # Generate a prediction with the generator
    predictions = gan_model(encoded_seq, training=False)

  # Select last predicted character
    predicted_char_index = tf.argmax(predictions, axis=-1)[:, -1].numpy()[0]
    predicted_char = int_to_char[predicted_char_index]

   # Add the predicted character to the generated sequence
    generated_text += predicted_char
    seed_text = generated_text[-seq_length:]

print("Generated Text:\n", generated_text)

