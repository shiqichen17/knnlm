import numpy as np
arr2 = np.random.randn(10,1)
knn= np.array([[1,2,3,4],[2,3,4,5]])
print(arr2)
print(arr2[knn])
print(arr2[knn].shape)