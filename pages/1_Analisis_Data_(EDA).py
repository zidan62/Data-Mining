import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("personality_dataset.csv")

df = load_data()

# Judul halaman
st.title("Distribusi Kategori Body Type")

# Plot distribusi Body Type
fig, ax = plt.subplots()
sns.countplot(data=df, x='Body Type', ax=ax)
plt.xticks(rotation=45)
ax.set_title('Distribusi Kategori Body Type')
st.pyplot(fig)

# Info tambahan
st.write("\nJumlah masing-masing kategori:")
st.dataframe(df['Body Type'].value_counts())
