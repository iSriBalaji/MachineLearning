import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

a=os.path.join(os.path.dirname(__file__),"train.csv")
#Reading file and removing missing data for all columns
data=pd.read_csv(os.path.join(os.path.dirname(__file__),"train.csv"))
data.dropna(axis=0)

features=["LotArea","OverallQual","OverallCond","YearBuilt"]
y=data.SalePrice.values
x=data[features].values

trainx,testx,trainy,testy=train_test_split(x,y,random_state=0)

regress=LinearRegression()
regress.fit(trainx,trainy)
predicted=regress.predict(testx)

error=mean_squared_error(testy,predicted)
print("Error",error)
print("t0:",regress.intercept_)
print("coeff",regress.coef_)

final_prediction=regress.intercept_+(x.dot(np.array(regress.coef_)))
with open("pre_result.txt","w") as f:
    for k,m in zip(final_prediction,y.reshape(-1,1)):
        f.write(str(int(k))+" "+str(m))
        f.write("\n")
    f.close()

myin=regress.predict(np.array([8246,5,8,1968]).reshape(1,4))
print("Prediction",myin)
print("--------------------------------------------------------------")
plt.title("House Price Prediction")
plt.xlabel("Year Built")
plt.ylabel("Price")
plt.scatter(x[:,3],y,label="Values",color="blue")
plt.plot([min(x[:,3]),max(x[:,3])],[min(final_prediction),max(final_prediction)],label="Regression Line",color="red")
plt.show()
