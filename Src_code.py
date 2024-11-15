import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
import pickle

df=pd.read_csv('D:/Updated Minor/fraud_detection.csv')
(df.head())

(df.isnull().sum())

(df.shape)

(df.type.value_counts())

type=df['type'].value_counts()

transactions=type.index

quantity=type.values

px.pie(df,values=quantity,names=transactions,hole=0.4,title="Distribution of Transaction Type")

df=df.dropna() 
(df)

df['isFraud']=df['isFraud'].map({0:'No Fraud',1:'Fraud'})
(df)

(df['type'].unique())
#df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],value=[2,4,1,3,5],inplace=True)
(df['type'].value_counts())
(df)

df['type']=df['type'].map({'PAYMENT':1, 'TRANSFER':4, 'CASH_OUT':2, 'DEBIT':5, 'CASH_IN':3})

(df['type'].value_counts())
(df)

(df['type'].unique())

(df['type'].value_counts())

x=df[['type','amount','oldbalanceOrg','newbalanceOrig']]
y=df.iloc[:,-2]

(y)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=42)

model=DecisionTreeClassifier()
(model.fit(xtrain,ytrain))

(model.score(xtest,ytest))  #Model completed

'''Ensure the input is reshaped to 2D (1 row, 4 columns in this case)'''
prediction = model.predict([[4,181,181,0]])
print(prediction)


'''Save the trained model as a .pkl file
with open("D:/Updated Minor/fraud_detection_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
'''

'''Load the model to check it was saved correctly
with open("D:/Updated Minor/fraud_detection_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)
'''
