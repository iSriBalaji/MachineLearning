import matplotlib.pyplot as plt
import numpy as np
# Link - https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
# refer the github link too

#dataset from geeksforgeek

input_data=[0, 1, 2, 3, 4, 5, 6, 7, 8]
output_data=[1, 3, 2, 5, 7, 8, 8, 9, 10]
x=np.array(input_data)
y=np.array(output_data)                             #creating np arrays


t0=1
t1=1
alpha=0.01
m=float(len(x))
epochs=1000


for i in range(epochs):
    predicted_result=t0+(t1*x)
    diff0=sum(y-predicted_result)/(-2*m)
    diff1=sum((y-predicted_result)*x)/(-2*m)
    t0=t0-alpha*diff0
    t1=t1-alpha*diff1

print(t0,t1)

final_fitted_data=t0+(t1*x)
print(t0+(t1*9))
plt.scatter(x,y)
plt.plot([min(x),max(x)],[min(final_fitted_data),max(final_fitted_data)],color='red')
plt.show()
