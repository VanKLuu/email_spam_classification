import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the CSV file
data = pd.read_csv('emails.csv')
labels = data.iloc[:, -1]
features = data.iloc[:, 1:-1]  # Exclude the first column (Email name) and the last column (labels)
feature_words = data.columns[1:-1]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert DataFrames and Series to numpy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Normalize data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Model architecture
model = Sequential([
    # Reshape input for Conv1D: Input shape should be (batch_size, steps, input_dim)
    tf.keras.layers.Reshape((3000, 1), input_shape=(3000,)),
    Conv1D(64, 3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Output the model's performance
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

# Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

plot_model(model, to_file='cnn_architecture.png', show_shapes=True)

fig = plt.figure(figsize=(15,4))

fig.add_subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training accuracy', 'Validation accuracy'])
plt.title('Training and validation accuracy')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')

fig.add_subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training loss', 'Validation loss'])
plt.title('Training and validation loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()


# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Calculate area under curve (AUC)
auc_score = auc(recall, precision)

# Plot precision-recall curve
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


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
    word_list = email_text.lower().split()
    word_counts = {word: word_list.count(word) for word in word_list}
    # Ensure generating a feature vector of the correct size
    email_vector = np.array([word_counts.get(word, 0) for word in feature_words])

    # Normalize the feature vector
    email_vector = email_vector / np.linalg.norm(email_vector) if np.linalg.norm(email_vector) != 0 else email_vector

    # Reshape for the model
    email_vector = email_vector.reshape(1, -1)

    # Predict using the model
    prediction = model.predict(email_vector)
    predicted_label = 'Spam' if prediction[0, 0] > 0.5 else 'Not Spam'
    return render_template('index.html',
                           prediction_text=f'This email is {predicted_label} with a probability of {prediction[0][0]:.2f}')


if __name__ == '__main__':
    app.run()
