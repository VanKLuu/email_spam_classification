import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the CSV file
file_path = 'emails.csv'
df = pd.read_csv(file_path)

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Ensure 'Prediction' column exists in the DataFrame
if 'Prediction' not in df.columns:
    raise ValueError("The 'Prediction' column does not exist in the CSV file.")

# Split the data into features (X) and target (y)
X = df.iloc[:, 1:-1]
y = df['Prediction']

# Feature selection using chi-square test
best_features = SelectKBest(score_func=chi2, k=1000)
X = best_features.fit_transform(X, y)

# Tokenize and vectorize the email content
max_words = 10000
max_sequence_length = 200
# Convert NumPy array to list of strings
X_texts = [' '.join(map(str, row)) for row in X]

# Tokenize and vectorize the email content
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_texts)
sequences = tokenizer.texts_to_sequences(X_texts)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Shuffle and split the data into training and testing sets
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Define Flask app
app = Flask(__name__)


# Route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    sequence = tokenizer.texts_to_sequences([email_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequence)[0][0]
    prediction_text = "Spam" if prediction > 0.5 else "Not Spam"
    return render_template('index.html',
                           prediction_text=f'This email is {prediction_text} with a probability of {prediction:.2f}')


if __name__ == '__main__':
    app.run()
