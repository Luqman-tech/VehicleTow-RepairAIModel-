from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key in production

import os

# Load the trained model and label encoders
base_path = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_path, 'model.joblib'))
le_location = joblib.load(os.path.join(base_path, 'le_location.joblib'))
le_problem = joblib.load(os.path.join(base_path, 'le_problem.joblib'))

def preprocess_input(request_time_str, vehicle_age, location, problem_description):
    try:
        # Parse Request_Time to datetime
        request_time = pd.to_datetime(request_time_str)
        hour = request_time.hour
        day = request_time.dayofweek
    except Exception:
        raise ValueError("Invalid date/time format for Request_Time.")

    # Encode categorical variables using loaded LabelEncoders
    location_code = le_location.transform([location]) if location in le_location.classes_ else -1  # Handle unseen locations

    problem_code = le_problem.transform([problem_description]) if problem_description in le_problem.classes_ else -1  # Handle unseen problems

    # Prepare feature array
    features = np.array([[hour, day, vehicle_age, location_code, problem_code]])
    return features

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_time = request.form.get('Request_Time')
        vehicle_age = request.form.get('Vehicle_Age')
        location = request.form.get('Location')
        problem_description = request.form.get('Problem_Description')

        # Validate inputs
        if not request_time or not vehicle_age or not location or not problem_description:
            flash("All fields are required.")
            return redirect(url_for('index'))

        vehicle_age = float(vehicle_age)

        # Preprocess input
        features = preprocess_input(request_time, vehicle_age, location, problem_description)

        # Predict
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=round(prediction, 2),
                               request_time=request_time,
                               vehicle_age=vehicle_age,
                               location=location,
                               problem_description=problem_description)

    except ValueError as ve:
        flash(str(ve))
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"An error occurred during prediction: {str(e)}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
