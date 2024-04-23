import pandas as pd
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from flask import Flask, request, render_template
from keras.models import Sequential
from keras.layers import Dense, Input

# Load the CSV file
file_path = 'emails.csv'
df = pd.read_csv(file_path)

# Ensure 'Prediction' column exists in the DataFrame
if 'Prediction' not in df.columns:
    raise ValueError("The 'Prediction' column does not exist in the CSV file.")

# Split the data into features (X) and target (y)
X = df.iloc[:, 1:-1]
y = df['Prediction']

# Split the data into training and testing sets
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reset the index of y_train
y_train = y_train.reset_index(drop=True)

# Define a custom stop words list
custom_stop_words = ["subject"] + list(text.ENGLISH_STOP_WORDS)


# Define a custom tokenizer
def custom_tokenizer(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in custom_stop_words]


# Convert the email content to TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=3000, tokenizer=custom_tokenizer)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text.astype('U'))
X_test_tfidf = tfidf_vectorizer.transform(X_test_text.astype('U'))

# Build the neural network model
nn_model = Sequential([
    Input(shape=(X_train_tfidf.shape[1],)),
    Dense(300, activation='relu'),
    Dense(1, activation='sigmoid')  # Use sigmoid activation for binary classification
])

# Compile the model with adjusted class weight
class_weight = {0: 1., 1: 5.}  # Assign higher weight to the minority class
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = nn_model.fit(X_train_tfidf.toarray(), y_train, epochs=50, batch_size=128, validation_split=0.2,
                       class_weight=class_weight)

# Train the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train[:X_train_tfidf.shape[0]])


# Evaluate the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype("int32").flatten()
    print("Accuracy:", accuracy_score(y_test, y_pred[:len(y_test)]))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred[:len(y_test)]))
    print("\nClassification Report:\n", classification_report(y_test, y_pred[:len(y_test)], zero_division=1))


print("\nNeural Network Classifier Results:")
evaluate_model(nn_model, X_test_tfidf.toarray(), y_test)

print("\nMultinomial Naive Bayes Classifier Results:")
evaluate_model(nb_classifier, X_test_tfidf, y_test)


def preprocess_email(email_content):
    email_features = tfidf_vectorizer.transform([email_content]).toarray()
    return email_features.reshape(1, -1)


# Classify the email using the neural network model
# def classify_email(model, email_content):
#     email_features = preprocess_email(email_content)
#
#     # Predict the probability of the email being spam
#     predicted_probability = model.predict(email_features)[0][0]
#
#     threshold = 0.5
#     prediction = "Spam" if predicted_probability > threshold else "Not Spam"
#
#     return prediction, predicted_probability


# Classify the email using the Naive Bayes Classifier
def classify_email(model, email_content):
    email_features = preprocess_email(email_content)

    # Predict the probability of the email being spam using the MultinomialNB classifier
    predicted_probability = model.predict_proba(email_features)[0][1]

    threshold = 0.4
    prediction = "Spam" if predicted_probability > threshold else "Not Spam"

    return prediction, predicted_probability


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form.get('email_text')

    # Classify the email using the Naive Bayes Classifier
    prediction, predicted_probability = classify_email(nb_classifier, email_text)

    # Classify the email using the neural network model
    # prediction, predicted_probability = classify_email(nn_model, email_text)

    return render_template('index.html',
                           prediction_text=f'This email is {prediction} with a probability of {predicted_probability:.2f}')


if __name__ == '__main__':
    app.run(debug=True)
