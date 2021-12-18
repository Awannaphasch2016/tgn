import numpy as np
from utils.utils import get_different_edges_mask_left

A = np.array([[1,4],[2,5],[3,6],[3,5]])
B = np.array([[1,4],[3,6],[7,8]])

A_nrows, A_ncols = A.shape
B_nrows, B_ncols = B.shape
A_dtype={'names':['f{}'.format(i) for i in range(A_ncols)],
       'formats':A_ncols * [A.dtype]}
B_dtype={'names':['f{}'.format(i) for i in range(B_ncols)],
       'formats':B_ncols * [B.dtype]}

# print(np.isin(A.view(A_dtype), B.view(B_dtype),invert=True))
# print(np.isin(B.view(B_dtype), A.view(A_dtype),invert=True))
# C = np.intersect1d(A.view(A_dtype), B.view(B_dtype))
# C = C.view(A.dtype).reshape(-1, ncols)


print(get_different_edges_mask_left(A, B))
# print(A[get_different_edges_mask_left(A, B)[0]])
