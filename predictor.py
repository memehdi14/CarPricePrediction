import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading the saved model, scaler, and encoders
knn = joblib.load("knn_car_price_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Function for depreciation
def apply_depreciation(predicted_price, vehicle_age):
    if vehicle_age == 0:
        return predicted_price
    elif vehicle_age <= 3:
        return predicted_price * 0.75
    elif vehicle_age <= 5:
        return predicted_price * 0.60
    elif vehicle_age <= 10:
        return predicted_price * 0.40
    else:
        return predicted_price * 0.30

# Prediction function
def predict_car_price(brand_name, model_name, vehicle_age, km_driven):
    if brand_name not in label_encoders['brand'].classes_ or model_name not in label_encoders['model'].classes_:
        return "Invalid brand or model."

    brand_encoded = label_encoders['brand'].transform([brand_name])[0]
    model_encoded = label_encoders['model'].transform([model_name])[0]

    input_data = np.array([[brand_encoded, model_encoded, vehicle_age, km_driven]])
    input_data = scaler.transform(input_data)

    predicted_log_price = knn.predict(input_data)[0]
    predicted_price = np.expm1(predicted_log_price)

    final_price = apply_depreciation(predicted_price, vehicle_age)

    return f"Estimated Price: ₹{final_price:,.2f}"

# Streamlit UI
st.title("Car Price Estimator")
st.subheader("Know Your Car’s Worth Before You Sell!")
st.write("Enter the details to get an estimated resale value.")

brand = st.selectbox("Car Brand", label_encoders['brand'].classes_)
model = st.selectbox("Car Model", label_encoders['model'].classes_)
vehicle_age = st.slider("Vehicle Age (Years)", 0, 15, 5)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)

if st.button("Estimate Price"):
    result = predict_car_price(brand, model, vehicle_age, km_driven)
    st.write(result)
