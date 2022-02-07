import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#reading the data...Creating a Dataframe
tit_data=pd.read_csv(r"C:\Users\Sri Balaji\Desktop\Sri Tut\Machine Learning\Titanic_Kaggle\train.csv")    #giving location as raw string to escape some characters
# print(tit_data.columns)
tit_data.dropna(axis=1)                                              #droping data which is unavailable

survived=tit_data.Survived                                           #for y
#param_for_train=['Pclass','Sex','Age','Cabin','Embarked']
param_for_train=['Pclass','Age']
tit_parameters=tit_data[param_for_train]                              #for X


train_in,test_in,train_out,test_out=train_test_split(tit_parameters,survived,random_state=1)     #Splitting Data

#creating Random Forest object
rf_model=RandomForestRegressor(random_state=1)
rf_model.fit(train_in,train_out)
