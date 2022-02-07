import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import  pandas as pd


#geeksforgeeks Data

input_data=[0, 1, 2, 3, 4, 5, 6, 7, 8]
output_data=[1, 3, 2, 5, 7, 8, 8, 9, 10]
x=pd.DataFrame(input_data)
y=pd.DataFrame(output_data)
regobj=LinearRegression()
regobj.fit(x,y)
y_result=regobj.predict(pd.DataFrame([9]))
print(y_result)