import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

st.title("Hasil Pelatihan Model")

df = pd.read_csv("personality_dataset.csv")

X = df[['Gender', 'Height', 'Weight']]
y = df['Index']

X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

st.write("**Akurasi:**", accuracy_score(y_test, predictions))
st.text("Classification Report:")
st.text(classification_report(y_test, predictions))

# Simpan model
with open("model_rf.pkl", "wb") as file:
    pickle.dump(model, file)
