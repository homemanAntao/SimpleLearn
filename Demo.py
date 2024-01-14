import numpy as np
from scipy import sparse

x = np.array([[1, 2, 3], [4, 5, 6]])
# print("x:\n{}".format(x))

# 稀疏矩阵 对角线为1 其余为0
eye = np.eye(4)
# print("numpy array:\n{}".format(eye))

# numpy数组转换为csr格式的scipy稀疏矩阵 :CSR (Compressed Sparse Row) 格式是一种常用的稀疏矩阵存储格式，它通过压缩稀疏矩阵的行索引、列索引和非零元素值的方式，
# 实现了对大规模稀疏矩阵的高效存储和读取。稀疏矩阵是指矩阵中大部分元素为零的矩阵，与之相对的是稠密矩阵，即大部分元素都不为零。
# CSR包含三个数组（所有的index从0开始）：
#
# V，用来存储矩阵中的非零元素的值；
# COL_INDEX，第i个元素记录了V[i]元素的列数；
# ROW_INDEX, 表示m个元素表示每一行第一个非零元素在数值中的索引
sparse_matrix = sparse.csr_matrix(eye)
# print(sparse_matrix)

data = np.ones(4)
# print(data)
row_index = np.arange(4)
print(row_index)
col_index = np.arange(4)
eye_co = sparse.coo_matrix((data, (row_index, col_index)))
print(eye_co)




