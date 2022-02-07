import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
a=os.path.join(os.path.dirname(__file__),"train.csv")

#Cost Function
def cost(h,y,m):
    return(sum((h-y)**2)/(2*m))

#Feature Scaling
def fea_scale(n):
    mea=n.mean()
    sd=n.std()
    return((n-mea)/sd)

#Reading file and removing missing data for all columns
data=pd.read_csv(os.path.join(os.path.dirname(__file__),"train.csv"))
data.dropna(axis=0)

y_data=data.SalePrice
y=np.array(y_data,dtype=np.float64)

x1_data=data["LotArea"]
x2_data=data["OverallQual"]
x3_data=data["OverallCond"]
x4_data=data["YearBuilt"]

m=len(y)
x0=np.ones(m)
x=np.array([x0,x1_data,x2_data,x3_data,x4_data],dtype=np.float64).T

#Applying Feature Scaling for x
for i in range(1,5):
    x[:,i]=fea_scale(x[:,i])

r,c=x.shape
theta=np.zeros(c)
alpha=0.01
epochs=6000

j_value=[]
ite=[]

#Gradient Descent
for k in range(1,epochs+1):
    h=x.dot(theta)          #calculating hypothesis function
    for i in range(0,5):
        theta[i]-=alpha*(sum((h-y)*x[:,i])/(2*m))    #Updating theta
    if(k%1000==0):
        print("epoch",k)
        h=x.dot(theta)
        j_value.append(cost(h,y,m))
        ite.append(k)

for i in range(0,5):
    print("theta",i,theta[i])

#Testing data
print("give input as: 9878 5 7 2006")
ls=list(map(int,input("Enter the 4 features: ").split()))
am=np.array(ls)
print(sum(np.insert(am,0,1)*theta))

#Visualizing cost function
plt.plot(ite,j_value)
plt.xlabel("No of Iterations")
plt.ylabel("Cost Function")
plt.title("Theta Values"+str(theta))
plt.show()

'''I suppose the initialising the theta value is the problem. As we were setting it to 1 it converge to a local minimum 
which is inefficient. So the theta should be initialised properly.But I don't know how to do it.'''

