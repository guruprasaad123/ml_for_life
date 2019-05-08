import numpy as np

a1=np.array([1,3])

print('a1',a1,a1.shape)

a2=np.dot(a1,np.array([[0,-1],[1,0]]))

print('transformation_matrix_1',a2,a2.shape)

a3=np.dot(a1,a2)

print(a3,('Non-Perpendicular','Perdicular')[a3==0])

a2=np.dot(np.array([[0,1],[-1,0]]),a1)

print('transformation_matrix_1',a2,a2.shape)

a3=np.dot(a1,a2)

print(a3,('Non-Perpendicular','Perdicular')[a3==0])


a4 = np.dot(np.array([1,4]),a1)

print(a4)