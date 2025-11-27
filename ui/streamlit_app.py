import os
import streamlit as st
import requests
from PIL import Image
import io

st.title("Plant Disease Classifier")

st.sidebar.info("Model v1 - Upload an image to predict")

# Use API_URL environment variable when deployed; fall back to localhost for local runs
API_URL = os.environ.get("API_URL", "http://localhost:8000")

uploaded_file = st.file_uploader("Upload an image", type=['jpg','png','jpeg'])
if uploaded_file:
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    res = requests.post(f"{API_URL}/predict", files=files)
    if res.ok:
        d = res.json()
        st.image(uploaded_file)
        st.write("Prediction:", d.get('class'))
        st.write("Probability:", d.get('probability'))
    else:
        st.error("Prediction failed: " + res.text)

# Retrain section
st.header("Retrain model")
zip_file = st.file_uploader("Upload zip with labeled folders for retrain", type=['zip'])
if zip_file:
    if st.button("Start Retrain"):
        files = {'file': (zip_file.name, zip_file.getvalue(), 'application/zip')}
        resp = requests.post(f"{API_URL}/retrain", files=files)
        st.write(resp.json())
