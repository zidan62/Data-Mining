import streamlit as st

st.set_page_config(page_title="Prediksi Kepribadian", layout="wide")

st.title("Selamat Datang di Dashboard Prediksi Kepribadian! 👋")

st.markdown("---")

st.header("Tentang Aplikasi Ini")
st.write("""
Aplikasi ini dirancang untuk memprediksi kategori tubuh seseorang (Underweight, Normal, Overweight, atau Obese) berdasarkan data BMI.

Dasbor ini terdiri dari tiga bagian utama:

1. **Analisis Data (EDA)**: Menjelajahi dataset yang digunakan, termasuk visualisasi distribusi data dan hubungan antar fitur.
2. **Hasil Pelatihan Model**: Menampilkan performa dari dua model machine learning (KNN dan Naive Bayes) yang telah dilatih untuk melakukan prediksi.
3. **Lakukan Prediksi**: Formulir interaktif di mana Anda dapat memasukkan data Anda sendiri dan mendapatkan hasil prediksi kategori tubuh.
""")
