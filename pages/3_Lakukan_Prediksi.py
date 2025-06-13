import streamlit as st
import pandas as pd
import pickle

# Judul halaman
st.title("Prediksi Kategori Tubuh 🧍")
st.write("Masukkan data BMI dan Gender untuk memprediksi kategori tubuh Anda.")

# Form input pengguna
with st.form("prediction_form"):
    gender = st.selectbox("Pilih Jenis Kelamin", ["Male", "Female"])
    height = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=170.0)
    weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=300.0, value=65.0)
    submit = st.form_submit_button("Prediksi")

# Load model
@st.cache_data
def load_model():
    with open("model_rf.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Proses prediksi
if submit:
    # Preprocessing input
    gender_binary = 1 if gender == "Male" else 0
    input_df = pd.DataFrame({
        "Gender": [gender_binary],
        "Height": [height],
        "Weight": [weight]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"Hasil Prediksi: {prediction}")
