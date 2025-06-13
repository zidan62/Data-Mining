import streamlit as st
import pandas as pd
import pickle

st.title("Prediksi Kategori Body Type")

@st.cache_data
def load_model():
    with open("model_rf.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

gender = st.selectbox("Pilih Gender", ["Male", "Female"])
height = st.number_input("Masukkan Tinggi Badan (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Masukkan Berat Badan (kg)", min_value=30, max_value=200, value=70)

if st.button("Prediksi"):
    input_data = pd.DataFrame({
        'Gender': [1 if gender == 'Male' else 0],
        'Height': [height],
        'Weight': [weight]
    })
    prediction = model.predict(input_data)[0]

    label_map = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obesity',
        5: 'Extreme Obesity'
    }

    st.success(f"Hasil Prediksi: {label_map[prediction]}")
