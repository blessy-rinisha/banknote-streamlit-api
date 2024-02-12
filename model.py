import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df=pd.read_csv("BankNote_Authentication.csv")

x=df.drop(columns=["class"])
y=df["class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)


#Saving model using pickle
pickle_out=open("model.pkl","wb")
pickle.dump(model,pickle_out)
pickle_out.close()

model.predict([[2,3,4,1]])
