import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import pickle
import tokenize

# Data Acquisition
def load_imdb_data(num_words=10000):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
    return (train_data, train_labels), (test_data, test_labels)

# Data Preprocessing
def preprocess_data(data, max_length=120):
    return pad_sequences(data, maxlen=max_length, padding='post', truncating='post')

# Load the data
(train_data, train_labels), (test_data, test_labels) = load_imdb_data()

# Preprocess the data
max_length = 120
train_data = preprocess_data(train_data, max_length)
test_data = preprocess_data(test_data, max_length)

# --- Test Case to Validate Data Preprocessing ---

# Helper function to decode a review
def decode_review(encoded_review, word_index):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    return decoded_review

def test_data_preprocessing(train_data, train_labels, word_index):
    # Check shapes
    print("Shape of training data:", train_data.shape)
    print("Shape of training labels:", train_labels.shape)

    # Assert conditions
    assert train_data.shape[0] == train_labels.shape[0], "Mismatch in number of samples between data and labels"
    assert train_data.shape[1] == max_length, "Incorrect sequence length after padding"

    # Visualize data distributions (length of reviews)
    lengths_before_padding = [len(review) for review in imdb.load_data()[0][0]]
    lengths_after_padding = [len(review) for review in train_data]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(lengths_before_padding, bins=30)
    plt.title("Lengths of Reviews Before Padding")
    plt.subplot(1, 2, 2)
    plt.hist(lengths_after_padding, bins=30)
    plt.title("Lengths of Reviews After Padding")
    plt.show()

    # Display a couple of samples
    for i in range(2):
        print("\nOriginal review:", decode_review(imdb.load_data()[0][0][i], word_index))
        print("Preprocessed review:", train_data[i])
        print("Label:", train_labels[i])

# Test Data Acquisition and Preprocessing
# if __name__ == "__main__":
#     word_index = imdb.get_word_index()
#     test_data_preprocessing(train_data, train_labels, word_index)

# Model Building
def build_model(vocab_size=10000, embedding_dim=16, max_length=120):
    model = Sequential([
        # Embedding Layer: To convert our word tokens (integers) into embeddings of fixed size
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        
        # LSTM Layer: Long Short Term Memory layer for capturing the sequence context
        LSTM(units=32, return_sequences=True),
        LSTM(units=16),

        # Dense Layer: Fully connected layer for classification
        Dense(units=6, activation='relu'),

        # Output Layer: Final layer with sigmoid activation for binary classification
        Dense(units=1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



import matplotlib.pyplot as plt

# Function to plot training and validation accuracy and loss
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plot training and validation accuracy
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    # Plot training and validation loss
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()



# Create the model
model = build_model()

# Model Summary
model.summary()




# Model Training and Evaluation

# Set training parameters
epochs = 10
batch_size = 32

# Splitting the training data into training and validation sets
validation_size = 0.2
validation_samples = int(len(train_data) * validation_size)
train_samples = len(train_data) - validation_samples

x_val = train_data[:validation_samples]
partial_x_train = train_data[validation_samples:]
y_val = train_labels[:validation_samples]
partial_y_train = train_labels[validation_samples:]

# Train the model
history = model.fit(partial_x_train, partial_y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val),
                    verbose=1)


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print("Test Accuracy:", test_acc)




# Plot the results
plot_history(history)


# Save the entire model to a HDF5 file
model.save('my_sentiment_model.h5')

# Save the history object as a pickle file
with open('my_training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)