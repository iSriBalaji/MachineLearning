import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

num=load_digits()
for x,y in zip(num.data[6:10],num.target[6:10]):
    plt.imshow(x.reshape(8,8),cmap="gray")
    plt.title(y)

