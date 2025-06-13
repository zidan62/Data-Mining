import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.header("Analisis Data (EDA)")

df = pd.read_csv("personality_dataset.csv")
st.dataframe(df.head())

st.subheader("Distribusi Kategori Body Type")
fig, ax = plt.subplots()
sns.countplot(x='Body Type', data=df, ax=ax)
st.pyplot(fig)

st.subheader("Korelasi antar Fitur")
fig2, ax2 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)
