import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('mode.chained_assignment', None)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(h,y,m):
    a=((y.T)*(np.log(h)))+(((1-y).T)*(np.log(1-h)))
    return (a/(-1*m))

te_path=os.path.join(os.path.dirname(__file__),"train.csv")
tr_path=os.path.join(os.path.dirname(__file__),"test.csv")

test_data=pd.read_csv(te_path)
train_data=pd.read_csv(tr_path)


xlist=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
train_x=train_data[xlist]
train_x["Bias"]=1
train_y=train_data["Chance of Admit "]

test_x=test_data[xlist]
test_x["Bias"]=1
test_y=test_data["Chance of Admit "]

x=test_x.values
y=test_y.values.reshape(-1,1)


m,n=x.shape
theta=np.ones((n,1),dtype=np.float64)
alpha=0.1
epochs=10000

cs=[]
epo=[]

for i in range(1,epochs+1):
    diff=(x.T).dot(sigmoid(x.dot(theta))-y)
    theta=theta-((alpha/m)*diff)
    if epochs%100==0:
        J=cost(sigmoid(x.dot(theta)),y,m)
        cs.append(J)
        epo.append(i)
print(theta)

# plt.plot(epo,cs)
# plt.show()