import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

st.header("Lakukan Prediksi")

height = st.slider("Masukkan Tinggi Badan (cm):", 100, 250, 170)
weight = st.slider("Masukkan Berat Badan (kg):", 30, 150, 60)

if st.button("Prediksi"):
    df = pd.read_csv("personality_dataset.csv")
    le = LabelEncoder()
    df['Body Type'] = le.fit_transform(df['Body Type'])

    X = df[['Height', 'Weight']]
    y = df['Body Type']

    model = GaussianNB()
    model.fit(X, y)

    pred = model.predict(np.array([[height, weight]]))
    result = le.inverse_transform(pred)[0]

    st.success(f"Kategori Tubuh Anda: **{result}**")
