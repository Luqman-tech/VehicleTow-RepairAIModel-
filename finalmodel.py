# -----------------------------------------
# Enhanced AI for SDG 9: Smarter Auto Repair Infrastructure
# Objective: Predict mechanic availability using AI (Refactored + Improved)
# -----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Step 1: Load and preprocess data
print("\n📥 Loading data...")
df = pd.read_csv("Motor_Vehicle_Repair_and_Towing.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('#', 'num')

# Drop rows with critical missing values
df.dropna(subset=['num_ase_certified_mechanics', 'tow_storage_zip', 'tow_storage_state'], inplace=True)

# Fill non-critical fields
df['city'] = df['city'].fillna('Unknown')
df['state'] = df['state'].fillna('MD')
df['tow_storage_address'] = df['tow_storage_address'].fillna('')

# Preview the cleaned data
print("\n📋 Preview of Cleaned Data:")
print(df.head().to_string(index=False))

# Step 2: Feature Engineering
print("\n🧠 Engineering features...")
df['tow_storage_zip'] = df['tow_storage_zip'].astype(str).str.extract(r'(\d+)')[0].astype(float)
df['address_length'] = df['tow_storage_address'].apply(lambda x: len(str(x)))

# Target variable
y = df['num_ase_certified_mechanics']

# Define preprocessing steps
text_features = ['tow_storage_address']
categorical_features = ['tow_storage_state', 'city']
numeric_features = ['tow_storage_zip', 'address_length']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        ('txt', TfidfVectorizer(max_features=20, stop_words='english'), 'tow_storage_address')
    ],
    remainder='drop'
)

# Step 3: Model pipeline with hyperparameter tuning
model_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20]
}

print("\n🔍 Starting grid search...")
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)

# Split data
X = df[['tow_storage_zip', 'tow_storage_state', 'city', 'address_length', 'tow_storage_address']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\n🚀 Training model...")
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
print(f"\n✅ Best Parameters: {grid_search.best_params_}")

# Step 4: Predictions and Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 5: Visual Insights
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\n🔍 Sample Predictions:")
print(comparison.head().to_string(index=False))

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='green')
plt.title("Actual vs Predicted Certified Mechanics")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

from flask import Flask

app = Flask(__name__)
if __name__ == "__main__":
    app.run()


import joblib

# Save the best model from GridSearch
joblib.dump(model, 'model/finalmodel.pkl')

