import streamlit as st
import pandas as pd
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("model_rf.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("Prediksi Kategori Body Type")

# Input pengguna
height = st.number_input("Masukkan tinggi badan (cm):", min_value=100, max_value=250, value=170)
weight = st.number_input("Masukkan berat badan (kg):", min_value=30, max_value=200, value=65)

# Lakukan prediksi
if st.button("Prediksi"):
    input_data = pd.DataFrame({
        'Height': [height],
        'Weight': [weight]
    })
    prediction = model.predict(input_data)
    st.success(f"Prediksi Body Type: {prediction[0]}")
