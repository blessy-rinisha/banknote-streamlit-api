import streamlit as st
import requests
from pydantic import BaseModel
import json

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

def predict_banknote(data: BankNote):
    response = requests.post("http://127.0.0.1:8000/predict", json=data.dict())
    result= response.json()

    prediction_value = result["prediction"]

    return prediction_value


st.title("Banknote Authentication")
st.text("Enter the values of the given data to check whether the BankNote💵 is fake or not😊!")

variance = st.number_input("Variance")
skewness = st.number_input("Skewness")
curtosis = st.number_input("Curtosis")
entropy = st.number_input("Entropy")

if st.button("Predict"):
    st.balloons()
    banknote = BankNote(variance=variance, skewness=skewness, curtosis=curtosis, entropy=entropy)
    prediction = predict_banknote(banknote)
    st.write(prediction)
    
