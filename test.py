# 删除DataFrame中的drop

import pandas as pd
import numpy as np
a = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
df = pd.DataFrame(a)
print(df)
train_data = np.array(df)
print(train_data)
# a.drop(0)  # 删除第0行，且不保存在a，即原a数组不被替换
x = df.drop(0, axis=1)  # 删除第0列，
train_data = np.array(x)
print(train_data)
#
# a.drop(0, axis=1, inplace=True)  # 删除列，返回的新数组被替换，保存在a中
# # aa = a.drop(0, axis=1)  # 同上，不过在被替换时，保存在aa 的新数组中
# print(a)

