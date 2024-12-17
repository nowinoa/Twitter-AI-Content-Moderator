import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the preprocessed dataset using pandas 
try:
    df = pd.read_csv('preprocessed_hate_speech_dataset.csv')
except FileNotFoundError:
    print("Error: Preprocessed dataset not found. Ensure the file exists in the current directory.")
    exit()

# Extract tweet and label columns
X = df['tweet'].values
y = df['class'].values

# Tokenize and pad sequences - divide the text into small sets and converts it into numercial data
max_len = 100  # max length of input sequences

# creates a dictionary with the tokenized words
tokenizer = Tokenizer()
# fits the tokenizer into the tweets to learn them
tokenizer.fit_on_texts(X) 

# maximum words for the dictionary = all tokenized words (the whole excel)
max_words = len(tokenizer.word_index) + 1 

# converts the text data into sequences of integer, each word represents an integer
X = tokenizer.texts_to_sequences(X)
# pad the sequences so they have all the same length
X = pad_sequences(X, maxlen=max_len)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    # turns words into 128-dimensional vectors
    Embedding(max_words, 128, input_length=max_len),  
    # learns layer to learn patterns in sequences
    LSTM(64, return_sequences=True),                 
    # reduces data size
    GlobalMaxPooling1D(),                            
    # hidden layer
    Dense(64, activation='relu'),                    
    Dropout(0.5),                                    
    # output layer
    Dense(3, activation='softmax')                   
])

# Compile the model selecting optimizer, loss function and tracking accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Function to classify new comments
def classify_comment(comment, model, tokenizer, max_len):
    '''Converts the comment into a sequence of numbers,
        pads the sequence to the max length, predict probabilities
        for each class (0,1,2) and returns the label with highest probability
        
        Args:
    - comment (str): the user input pretending it is a tweet
    - model (keras.Model): trained machine learning model for classification
    - tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer used during training to preprocess text
    - max_len (int): max length of input sequences (used for padding)

    Returns:
    - label (str): The predicted label for the input comment ('hate speech', 'offensive speech', or 'accepted speech').
    - probabilities (numpy.ndarray): Probabilities for each class.'''
    
    seq = tokenizer.texts_to_sequences([comment])
    padded_seq = pad_sequences(seq, maxlen=max_len)
    probabilities = model.predict(padded_seq)[0]
    label = np.argmax(probabilities)
    
    labels = {0: 'hate speech', 1: 'offensive speech', 2: 'accepted speech'}
    return labels[label], probabilities

# user input
print("\nEnter a comment to classify (or type 'quit' to exit):")
while True:
    comment = input()
    if comment.lower() == 'quit':
        break
    label, probs = classify_comment(comment, model, tokenizer, max_len)
    print(f"\nPredicted Label: {label}")
    print(f"Probabilities: {probs}")

