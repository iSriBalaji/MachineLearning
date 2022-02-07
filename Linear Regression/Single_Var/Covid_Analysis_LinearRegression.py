import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
plt.style.use("ggplot")
input_data=list(range(1,24))
output_data=[2303,2554,2720,3865,2962,3552,3379,3370,3099,4370,3619,3610,3700,3880,3760,4874,4975,4713,5821,4680,3435,6088,6654]

x=np.array(input_data)
y=np.array(output_data)

theta_0=2000
theta_1=100
alpha=0.01
m=float(len(input_data))
epochs=10000

for i in range(epochs):
    prediction=theta_0+(theta_1*x)
    diff_0=(sum(output_data-prediction))/(-m)
    diff_1=(sum((output_data-prediction)*x))/(-m)
    theta_0=theta_0-(alpha*diff_0)
    theta_1=theta_1-(alpha*diff_1)
final_fit=theta_0+(theta_1*x)

print("Parameters used were: ",theta_0,theta_1)

print("May 24:",round(theta_0+(theta_1*24)))


plt.scatter(x,y,color="#007788",label="Cases")
plt.plot([min(x),max(x)],[min(final_fit),max(final_fit)],color="#CC2A49",label="Regression Line")
plt.title("COVID-19 Analysis (May,2020)")
plt.xlabel("Date")
plt.ylabel("No of Cases")
plt.legend()
plt.tight_layout()
plt.show()


