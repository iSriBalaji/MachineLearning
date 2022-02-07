import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(z))

te_path=os.path.join(os.path.dirname(__file__),"test.csv")
tr_path=os.path.join(os.path.dirname(__file__),"train.csv")

test_data=pd.read_csv(te_path)
train_data=pd.read_csv(tr_path)
# test_data.dropna(axis=0)
# train_data.dropna(axis=0)
x_list=['PassengerId','Pclass','Sex', 'Age', 'SibSp', 'Parch','Fare','Embarked']
train_x=train_data[x_list]
train_x["Bias"]=1
train_y=train_data['Survived']
test_x=test_data[x_list]
test_x["Bias"]=1
# test_y=test_data['Survived']

na=train_x.isnull().sum()
print(na)

age_medi=train_x['Age'].median()
train_x['Age']=train_x['Age'].fillna(age_medi)

age_medi=test_x['Age'].median()
test_x['Age']=test_x['Age'].fillna(age_medi)

fare=test_x['Fare'].mean()
test_x['Fare']=test_x['Fare'].fillna(fare)

em_mode=train_x["Embarked"].mode()
train_x["Embarked"].fillna(em_mode)
print(train_x["Embarked"].value_counts())
emb={"Embarked":{"S":1,"C":2,"Q":3}}
train_x.replace(emb,inplace=True)
#train_x=train_x.replace(emb)

em_mode=test_x["Embarked"].mode()
test_x["Embarked"].fillna(em_mode)
print(test_x["Embarked"].value_counts())
emb={"Embarked":{"S":1,"C":2,"Q":3}}
test_x.replace(emb,inplace=True)

train_x=pd.get_dummies(train_x,columns=["Sex"],drop_first=True,dtype=np.int64)
test_x=pd.get_dummies(test_x,columns=["Sex"],drop_first=True,dtype=np.int64)
print(test_x.head(10))

tr_x=train_x.to_numpy()
tr_y=train_y.values.reshape(-1,1)
alpha=0.1
m,n=tr_x.shape
print(m,n)
print(tr_y.shape)
# theta=np.zeros((n,1),dtype=np.int64)
theta=np.array([10,10,10,10,10,10,10,10,10]).reshape(-1,1)
print("theta",theta.shape)
print(train_x.dtypes)

epochs=1
#Vectorized Implementation
for i in range(1,epochs+1):
    diff=(tr_x.T).dot(sigmoid(tr_x.dot(theta))-tr_y)
    theta=theta-((alpha/m)*diff)
print(diff)

# em_mode=train_x["Embarked"].mode()
# train_x["Embarked"]=train_x["Embarked"].fillna(em_mode)