import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import joblib

def train_and_save_model():
    import os
    # Load Dataset
    base_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_path, "Motor_Vehicle_Repair_and_Towing.csv"))

    # Data Cleaning
    features_to_check = ['tow_storage_zip', 'tow_storage_address', 'tow_storage_state', 'num_ase_certified_mechanics']
    df = df.dropna(subset=features_to_check)
    df['city'] = df['city'].fillna('Unknown')
    df['state'] = df['state'].fillna('MD')  # Assuming most entries are from Maryland

    # Preprocess Data
    df['Hour'] = pd.to_datetime(df['Request_Time']).dt.hour
    df['Day'] = pd.to_datetime(df['Request_Time']).dt.dayofweek

    le_location = LabelEncoder()
    le_problem = LabelEncoder()
    df['Location_Code'] = le_location.fit_transform(df['Location'])
    df['Problem_Code'] = le_problem.fit_transform(df['Problem_Description'])

    # Feature selection
    features = ['Hour', 'Day', 'Vehicle_Age', 'Location_Code', 'Problem_Code']
    X = df[features]
    y = df['num_ase_certified_mechanics']  # Assuming this is the target variable

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model and encoders in the script directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(base_path, 'model.joblib'))
    joblib.dump(le_location, os.path.join(base_path, 'le_location.joblib'))
    joblib.dump(le_problem, os.path.join(base_path, 'le_problem.joblib'))

    print("Model and encoders saved successfully.")

if __name__ == "__main__":
    train_and_save_model()

