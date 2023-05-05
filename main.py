"""
==============================
SMS Spam Detection Program
==============================
Written by Ethan D'Mello
December 12, 2022
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    GlobalMaxPool1D,
    Dense,
)
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data.txt", sep="\t", names=["label", "message"])

# Encode labels: ham=0, spam=1
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.3, random_state=42
)

# Tokenize messages
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_length = 100
X_train_padded = pad_sequences(
    X_train_sequences, maxlen=max_length, padding="post", truncating="post"
)
X_test_padded = pad_sequences(
    X_test_sequences, maxlen=max_length, padding="post", truncating="post"
)

# Define Bidirectional LSTM model
model = Sequential(
    [
        Embedding(max_words, 32, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile and train model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train_padded, y_train, epochs=10, validation_split=0.2, verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Save tokenizer and model
import pickle

with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model.save("spam_filter_bidirectional_lstm.h5")
