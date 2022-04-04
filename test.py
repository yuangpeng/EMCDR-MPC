import numpy as np
x = [[[1,2],[4,5],[7,8]],[[10,12],[14,15],[17,18]],[[10,12],[14,15],[17,18]],[[10,12],[14,15],[17,18]]]
# print(x.shape)
y = np.swapaxes(x,0,1)

print(y.shape)
y.tolist()
print(y)