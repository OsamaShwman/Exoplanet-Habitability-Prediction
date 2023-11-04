import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Create a DataFrame for the single row of data
data = {
    'pl_orbper': [4.91],
    'pl_mass': [1.05],
    'pl_rad': [1.02],
    'pl_orbeccen': [0],
    'st_teff': [2904],
    'st_rad': [0.11],
    'st_mass': [0.09],
    'st_logg': [8],
    'st_age': [0.95],
    'ESI': [0.8]
}

new_data = pd.DataFrame(data)

# Preprocess the new data with the same StandardScaler used for training data
scaler = joblib.load('scaler.pkl')
new_data_scaled = scaler.transform(new_data)

# Load the trained model
loaded_model = tf.keras.models.load_model('best_model.h5')

# Make predictions on the new data
new_predictions = loaded_model.predict(new_data_scaled)
new_predictions = (new_predictions > 0.5)  # Convert probabilities to binary values

# Add the predictions as a new column in the new_data DataFrame
new_data['ESI-ROUND'] = new_predictions

# Print the new_data DataFrame with predictions
print(new_data['ESI-ROUND'])