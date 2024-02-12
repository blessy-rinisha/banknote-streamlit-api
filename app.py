import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
from pydantic import BaseModel


#creating app object

app=FastAPI()
pickle_in=open("model.pkl","rb")
model=pickle.load(pickle_in)



class BankNote(BaseModel):
    variance:float
    skewness:float
    curtosis:float
    entropy:float



@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    prediction=model.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"

    return{
        'prediction': prediction
    }

#running created api using uvicorn

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)

    # uvicorn app:app --reload 