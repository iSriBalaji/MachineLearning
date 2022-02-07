import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
a=os.path.join(os.path.dirname(__file__),"train.csv")

def cost(h,y,m):
    pass

data=pd.read_csv(os.path.join(os.path.dirname(__file__),"train.csv"))
data.dropna(axis=0)

y_data=data.SalePrice
y=np.array(y_data,dtype=np.int64)


x1_data=data["LotArea"]
x2_data=data["OverallQual"]
x3_data=data["OverallCond"]
x4_data=data["YearBuilt"]

m=len(y)
x0=np.ones(m)
x=np.array([x0,x1_data,x2_data,x3_data,x4_data],dtype=np.int64).T

theta=np.ones(len(x))
alpha=0.01
epochs=1000

t=((np.linalg.inv((x.T).dot(x))).dot(x.T)).dot(y)
print(t)

fit=np.dot(x,t)

plt.scatter(x[:,4],y)
plt.plot([min(x[:,4]),max(x[:,4])],[min(fit),max(fit)],color="red")
#plt.show()

for i in range(0,5):
    print(t[i])

while(1):
    ls=list(map(int,input("Enter the 4 features: ").split()))
    am=np.array(ls)
    print(sum(np.insert(am,0,1)*t))
    break




