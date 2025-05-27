# Import necessary libraries
from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load the dataset
df = pd.read_csv('Gold Price (2013-2023).csv')

# Ensure 'Price' is treated as string to replace commas
df['Price'] = df['Price'].astype(str).str.replace(',', '')

# Convert 'Price' column to float
df['Price'] = df['Price'].astype(float)

# Parse and sort dates
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# Preprocessing
scaler = MinMaxScaler()
scaler.fit(df['Price'].values.reshape(-1, 1))
window_size = 60

# Load the model
model = tf.keras.models.load_model('GoldModel.keras')

# Prediction function
def predict_gold_price(date):
    date = pd.to_datetime(date)
    if date < df['Date'].min():
        index = 0
    elif date > df['Date'].max():
        index = df.shape[0] - window_size
    else:
        index = df[df['Date'] >= date].index[0]
    # Ensure the index is within bounds
    index = max(window_size, min(index, df.shape[0] - 1))
    input_data = df['Price'][index - window_size:index].values
    input_data = scaler.transform(input_data.reshape(-1, 1)).reshape(1, window_size, 1)
    prediction = model.predict(input_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0][0]

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the buy gold page
@app.route('/buygold')
def buygold():
    return render_template('buygold.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    user_date = request.form['date']
    predicted_price = predict_gold_price(user_date)
    return render_template('result.html', date=user_date, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
