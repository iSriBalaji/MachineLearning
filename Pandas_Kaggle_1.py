import pandas as pd
from sklearn.tree import DecisionTreeRegressor
val=pd.read_csv(r'C:\Users\Sri Balaji\Desktop\Sri Tut\Machine Learning\melb_data.csv')
x=val.columns
print(x)
p=val.Price
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# kal=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# x=val[kal]
# model=DecisionTreeRegressor(random_state=1)
# model.fit(x,p)
# result=model.predict(x.head())
# # print(help(DecisionTreeRegressor))


