# 🚗 Car Price Estimator

A machine learning-powered web application built using **Streamlit** that estimates the resale price of used cars based on brand, model, vehicle age, and kilometers driven.

---

## 📌 Project Overview

This project uses a **K-Nearest Neighbors (KNN)** regression model trained on the **CarDekho dataset** to predict the log-transformed selling price of used cars. The prediction is then adjusted using a depreciation function to provide a realistic resale value.

---

## 🔍 Features

- Predict resale value based on:
  - Car **brand** and **model**
  - **Vehicle age** (in years)
  - **Kilometers driven**
- Dynamic depreciation calculation based on vehicle age
- Clean and interactive Streamlit interface
- Pretrained model with label encoding and feature scaling

---

## 🧠 Machine Learning Model

- **Model**: KNN Regressor
- **Target Variable**: Log of car selling price
- **Preprocessing**:
  - `LabelEncoder` for categorical columns
  - `StandardScaler` for numerical features
- **Postprocessing**:
  - Exponentiation of predictions
  - Age-based depreciation adjustment

---

## 🗃️ Dataset

- Source: [CarDekho Dataset](./cardekho_dataset.csv)
- Features used:
  - `brand`, `model`, `vehicle_age`, `km_driven`
- Target:
  - `selling_price` (log-transformed during training)

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/memehdi14/carpriceprediction.git
   cd carpriceprediction
