from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("model/mechanic_model.pkl")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        zip_code = float(request.form['zip'])
        state = request.form['state']
        city = request.form['city']
        address = request.form['address']

        data = pd.DataFrame([{
            'tow_storage_zip': zip_code,
            'tow_storage_state': state,
            'city': city,
            'address_length': len(address),
            'tow_storage_address': address
        }])

        prediction = model.predict(data)[0]
        return render_template('index.html', prediction_text=f'Estimated Certified Mechanics: {round(prediction)}')

if __name__ == '__main__':
    app.run(debug=True)
