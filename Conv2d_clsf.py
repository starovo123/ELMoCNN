import numpy as np
lda_list = [[1,2,3],[4,5,6]]

lda = np.array(lda_list)
print(lda)
wordemb = np.array([[7,8,9,10,11], [12,13,14,15,16]])
print(wordemb)
matrix = np.concatenate((lda,wordemb),axis=1)
print(matrix)