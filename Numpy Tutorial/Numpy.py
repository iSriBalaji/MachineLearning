#NUMPY
#Multidimensional array library
''' List are slow and numpy is faster as it used fixed type usually int32 or they can also have other fixed types
so they take less space when compared to list'''
#we don't want to check the type of the element while iteration
#numpy saves element in contiguous or continuous memory whereas the elements in list where scatters in the memory and matched using pointers
#It's an alternative for MATLAB
#Useful for plotting(matplotlib)
#backend of pandas , digital photography
#Used in Machine Learning


import numpy as np

#inializing array
a=np.array([1,2,3,4,5],dtype="int64")
print(a)

#2d array
b=np.array([[2,3,4],[5,6,7]])
print(b)

#Getting dimension
print("Dimension of a",a.ndim)
print("Dimension of b",b.ndim)

#Getting Shape of the array
print("Shape of a",a.shape)
print("Shape of b",b.shape)

#Getting type of array
print(a.dtype)          #changed dtype of a at initializing but the default is int32
print(b.dtype)      

#Getting the size (bytes)
print(a.itemsize)   #int8 has 1 byte int16 has 2 bytes and it goes on
print(b.itemsize)   