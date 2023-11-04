from rest_framework.response import Response

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from rest_framework.decorators import api_view
import pandas as pd

@api_view(('POST',))
def api(request):
    data = {
        'pl_orbper': [request.POST.get('pl_orbper')],
        'pl_mass': [request.POST.get('pl_mass')],
        'pl_rad': [request.POST.get('pl_rad')],
        'pl_orbeccen': [request.POST.get('pl_orbeccen')],
        'st_teff': [request.POST.get('st_teff')],
        'st_rad': [request.POST.get('st_rad')],
        'st_mass': [request.POST.get('st_mass')],
        'st_logg': [request.POST.get('st_logg')],
        'st_age': [request.POST.get('st_age')],
        'ESI': [request.POST.get('ESI')]
    }

    new_data = pd.DataFrame(data)

    # Preprocess the new data with the same StandardScaler used for training data
    scaler = joblib.load('hackathon/AI/scaler.pkl')
    new_data_scaled = scaler.transform(new_data)

    # Load the trained model
    loaded_model = tf.keras.models.load_model('hackathon/AI/best_model.h5')

    # Make predictions on the new data
    new_predictions = loaded_model.predict(new_data_scaled)
    new_predictions = (new_predictions > 0.5)  # Convert probabilities to binary values

    # Add the predictions as a new column in the new_data DataFrame
    new_data['ESI-ROUND'] = new_predictions

    # Print the new_data DataFrame with predictions
    result = str(new_data['ESI-ROUND'].values[0])

    return Response(data={"ESI": f"{result}"}, status=200)