import string
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the cleaned Shakespeare text
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower().translate(str.maketrans("", "", string.punctuation))

# Tokenize the text at the character level
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences([text])[0]

# Create input-output pairs
sequence_length = 20
X, y = [], []

for i in range(sequence_length, len(sequences)):
    X.append(sequences[i-sequence_length:i])
    y.append(sequences[i])

X, y = np.array(X), np.array(y)
y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

print("Preprocessing completed successfully.")

# model degin

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Define the LSTM model
vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(vocab_size, 50, input_length=sequence_length),
    LSTM(128),  # Reduced LSTM units
    Dense(128, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# model training

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint("lstm_shakespeare.h5", save_best_only=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1, batch_size=1024,
    callbacks=[early_stopping, checkpoint]
)

print("Model training completed successfully.")

# text generation


# Text generation function
import numpy as np

# Batch generation function
def generate_text_batch(seed_text, next_chars=1000, batch_size=50):
    generated = seed_text
    for _ in range(next_chars // batch_size):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        if len(token_list) < sequence_length:
            token_list = np.pad(token_list, (sequence_length - len(token_list), 0), mode='constant')
        else:
            token_list = token_list[-sequence_length:]

        token_list = np.expand_dims(token_list, axis=0)

        # Predict batch of tokens
        predictions = model.predict(token_list, verbose=0)

        # Select top predictions
        for _ in range(batch_size):
            predicted = predictions[0]
            next_token = np.random.choice(len(predicted), p=predicted)
            
            for char, index in tokenizer.word_index.items():
                if index == next_token:
                    generated += char
                    seed_text += char
                    break

    return generated

# Generate batch text
seed = "shall i compare thee to a summer's day"
generated_text = generate_text_batch(seed, next_chars=1000, batch_size=100)
print("Generated Text:")
print(generated_text)
