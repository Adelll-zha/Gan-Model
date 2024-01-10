import tensorflow as tf
import numpy as np

# Charger les modèles préentraînés
generator = tf.keras.models.load_model('generator_model.keras')
discriminator = tf.keras.models.load_model('discriminator_model.keras')

# Créer un modèle combiné
gan_model = tf.keras.Sequential()
gan_model.add(generator)
gan_model.add(discriminator)

# Compiler le modèle combiné
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

# Définir la longueur de la séquence initiale et le nombre de caractères à générer
seq_length = 50
num_chars = 20

# Définir la séquence initiale
seed_text = "Hello"
# Charger les données et définir char_to_int, vocab_size, int_to_char
data = open('C:/Users/romai/OneDrive/Bureau/ai prise/bot chat/data.txt', 'r').read()
chars = list(set(data))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
seq_length = 50

# Générer du texte avec le modèle GAN
generated_text = seed_text
for _ in range(num_chars):
    # Ajuster la longueur de la séquence initiale
    current_seq = ' ' * (seq_length - len(seed_text)) + seed_text

    # Encoder la séquence
    encoded_seq = np.array([char_to_int[char] for char in current_seq])
    encoded_seq = np.eye(vocab_size)[encoded_seq.reshape(-1)]
    encoded_seq = encoded_seq.reshape(1, seq_length, vocab_size)

    # Générer une prédiction avec le générateur
    predictions = gan_model(encoded_seq, training=False)

    # Sélectionner le dernier caractère prédit
    predicted_char_index = tf.argmax(predictions, axis=-1)[:, -1].numpy()[0]
    predicted_char = int_to_char[predicted_char_index]

    # Ajouter le caractère prédit à la séquence générée
    generated_text += predicted_char
    seed_text = generated_text[-seq_length:]

print("Generated Text:\n", generated_text)

