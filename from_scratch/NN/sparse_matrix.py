from scipy.sparse import coo_matrix
import numpy as np

# Constructing a matrix using ijv format
row  = np.array([0, 3, 1, 0])
col  = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
coo = coo_matrix((data, (row, col)))


print(coo.shape[1])