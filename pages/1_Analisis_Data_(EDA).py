import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.title("Distribusi Kategori Body Type")

df = pd.read_csv("personality_dataset.csv")

df['Index'] = df['Index'].map({
    0: 'Extremely Weak',
    1: 'Weak',
    2: 'Normal',
    3: 'Overweight',
    4: 'Obesity',
    5: 'Extreme Obesity'
})

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Index', palette='viridis')
plt.xticks(rotation=45)
plt.title("Distribusi Body Type")
st.pyplot(plt)
