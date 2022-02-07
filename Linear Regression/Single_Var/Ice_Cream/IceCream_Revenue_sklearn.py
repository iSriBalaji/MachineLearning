import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
plt.style.use("ggplot")

pa=os.path.join(os.path.dirname(__file__),"IceCreamData.csv")

data=pd.read_csv(pa)
data.dropna(axis=0)

x=data["Temperature"].values
y=data["Revenue"].values

x=x.reshape(-1,1)
y=y.reshape(-1,1)

trainx,testx,trainy,testy=train_test_split(x,y,random_state=0)


regress=LinearRegression()
regress.fit(trainx,trainy)
predicted=regress.predict(testx)

error=mean_squared_error(testy,predicted)
print(error)
print(regress.intercept_)
print(regress.coef_)

final_prediction=regress.intercept_+(x*regress.coef_)

myin=regress.predict(np.array([40]).reshape(-1,1))
print(myin)

plt.title("IceCream Revenue Analysis")
plt.xlabel("Temperature")
plt.ylabel("Revenue")
plt.scatter(x,y,label="Values",color="blue")
plt.plot([min(x),max(x)],[min(final_prediction),max(final_prediction)],label="Regression Line",color="red")
plt.show()




