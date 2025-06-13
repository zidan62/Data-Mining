import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

st.header("Hasil Pelatihan Model")

df = pd.read_csv("personality_dataset.csv")
le = LabelEncoder()
df['Body Type'] = le.fit_transform(df['Body Type'])

X = df[['Height', 'Weight']]
y = df['Body Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

st.subheader("KNN Classification Report")
st.text(classification_report(y_test, y_pred_knn, target_names=le.classes_))

st.subheader("Naive Bayes Classification Report")
st.text(classification_report(y_test, y_pred_nb, target_names=le.classes_))
