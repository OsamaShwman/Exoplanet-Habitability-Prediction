import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import joblib

# Load the dataset
dataset = pd.read_csv('2468_cleaned-3.csv')

# Split the dataset into features (x) and target (y)
x = dataset.drop(columns=["ESI-ROUND"])
y = dataset["ESI-ROUND"]

# Split the data into a training set and a testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Data Preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build the model
model = models.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for monitoring and saving the best model
early_stopping = EarlyStopping(patience=100, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
tensorboard = TensorBoard(log_dir='logs')

# Train the model
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint, tensorboard])

# Load the best model
best_model = tf.keras.models.load_model('best_model.h5')

# Evaluate the best model on the test data
y_pred = best_model.predict(x_test)
y_pred = (y_pred > 0.5)  # Convert probabilities to binary values

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred,zero_division=1)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Save the fitted StandardScaler to a file
joblib.dump(scaler, 'scaler.pkl')
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)