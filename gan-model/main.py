import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import numpy as np

# Hyperparameters
print(tf.config.list_physical_devices())
num_units = 128
num_classes = 86 #256
batch_size = 64
seq_length = 50
learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
num_epochs = 1

# Data Preparation
data = open('C:/Users/romai/OneDrive/Bureau/ai prise/bot chat/data.txt', 'r').read()
chars = list(set(data))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)


X = np.zeros((len(data) // seq_length, seq_length, vocab_size), dtype=np.float32)
Y = np.zeros((len(data) // seq_length, seq_length, vocab_size), dtype=np.float32)
for i in range(0, len(data) // seq_length):
    X_sequence = data[i * seq_length:(i + 1) * seq_length]
    X[i] = np.eye(vocab_size)[[char_to_int[char] for char in X_sequence]]
    Y_sequence = data[i * seq_length + 1:(i + 1) * seq_length + 1]
    Y[i] = np.eye(vocab_size)[[char_to_int[char] for char in Y_sequence]]

# GAN Model
generator = tf.keras.Sequential([
    tf.keras.layers.LSTM(num_units, return_sequences=True, input_shape=(seq_length, vocab_size)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.LSTM(num_units, return_sequences=True, input_shape=(seq_length, num_classes)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta1, beta2, epsilon)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta1, beta2, epsilon)

# Training loop

for epoch in range(num_epochs):
    print("traning: ",  {epoch})
    for batch in range(len(data) // (seq_length * batch_size)):
        start_index = batch * batch_size
        end_index = start_index + batch_size
        X_batch = X[start_index:end_index]
        Y_batch = Y[start_index:end_index]

        # Train the generator
        with tf.GradientTape() as gen_tape:
            generated_data = generator(X_batch, training=True)
            gen_loss = cross_entropy(Y_batch, generated_data)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            real_output = discriminator(Y_batch, training=True)
            fake_output = discriminator(generated_data, training=False)
            disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Print losses
        if batch % 50 == 0:
            print(f'Epoch {epoch+1}, Batch {batch}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}')
        
        
            
print("end")
generator.save('generator_model.keras')
discriminator.save('discriminator_model.keras')