	import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

te_path=os.path.join(os.path.dirname(__file__),"test.csv")
tr_path=os.path.join(os.path.dirname(__file__),"train.csv")
te_res=os.path.join(os.path.dirname(__file__),"gender_submission.csv")
sub_file=os.path.join(os.path.dirname(__file__),"my_submission.csv")
#Turned off warning in pandas
pd.set_option('mode.chained_assignment', None)
#---------------------------------------------------------------------------------------------------------------------#
test_data=pd.read_csv(te_path)
train_data=pd.read_csv(tr_path)
test_res=pd.read_csv(te_res)

x_list=['PassengerId','Pclass','Sex', 'Age', 'SibSp', 'Parch','Fare','Embarked']
train_x=train_data[x_list]
train_y=train_data['Survived']
test_x=test_data[x_list]
test_y=test_res['Survived']
#----------------------------------------------------------------------------------------------------------------------#

#Getting Ready the Data for Modelling
age_medi=train_x['Age'].median()
train_x['Age']=train_x['Age'].fillna(age_medi)

age_medi=test_x['Age'].median()
test_x['Age']=test_x['Age'].fillna(age_medi)

fare=test_x['Fare'].mean()
test_x['Fare']=test_x['Fare'].fillna(fare)

em_mode=train_x["Embarked"].value_counts().index[0]
train_x["Embarked"]=train_x["Embarked"].fillna(em_mode)
emb={"Embarked":{"S":1,"C":2,"Q":3}}
train_x.replace(emb,inplace=True)

em_mode=test_x["Embarked"].value_counts().index[0]
test_x["Embarked"]=test_x["Embarked"].fillna(em_mode)
emb={"Embarked":{"S":1,"C":2,"Q":3}}
test_x.replace(emb,inplace=True)

train_x=pd.get_dummies(train_x,columns=["Sex"],drop_first=True,dtype=np.int64)
test_x=pd.get_dummies(test_x,columns=["Sex"],drop_first=True,dtype=np.int64)
# print(train_x.isnull().sum())
# print(test_x.isnull().sum())
# print(train_y.isnull().sum())
# print(test_res.isnull().sum())
#-------------------------------------------------------------------------------------------------------------------#
tr_x=train_x.values
tr_y=train_y.values
te_x=test_x.values
te_y=test_y.values

#Data is completely ready for modelling
model=RandomForestRegressor(random_state=0,max_depth=4)
model.fit(tr_x,tr_y)
prediction=model.predict(te_x)
predict=np.around(prediction).astype("int8")
error=mean_squared_error(te_y,predict)
print(error)

output = pd.DataFrame({'PassengerId': test_res.PassengerId, 'Survived': predict})
output.to_csv(sub_file, index=False)
print("Your submission was successfully saved!")
