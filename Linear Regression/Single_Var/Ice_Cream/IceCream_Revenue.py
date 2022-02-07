import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
plt.style.use("ggplot")
data=pd.read_csv(os.path.join(os.path.dirname(__file__),"IceCreamData.csv"))
#data=pd.read_csv(r"C:\Users\Sri Balaji\Desktop\Sri Tut\Machine Learning\Linear Regression\Ice_Cream\IceCreamData.csv")
data.dropna(axis=0)
#print(data.head())
x=data["Temperature"]
y=data["Revenue"]

theta_0=0
theta_1=0
alpha=0.001
m=float(len(x))
epochs=10000

st=time.time()
for i in range(epochs):
    prediction=theta_0+(theta_1*x)
    diff_0=sum(y-prediction)/(-2*m)
    diff_1=(sum((y-prediction)*x))/(-2*m)
    theta_0=theta_0-(alpha*diff_0)
    theta_1=theta_1-(alpha*diff_1)
    if(i%1000==0):
         print(time.time()-st)

print(r"Parameters used---t0 and t1:",theta_0,theta_1)
final_prediction=theta_0+(theta_1*x)
print(theta_0+(theta_1*40))
plt.title("IceCream Revenue Analysis")
plt.xlabel("Temperature")
plt.ylabel("Revenue")
plt.scatter(x,y,label="Values",color="blue")
plt.plot([min(x),max(x)],[min(final_prediction),max(final_prediction)],label="Regression Line",color="red")
plt.legend()
plt.show()
