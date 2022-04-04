import csv
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
Batch_size = 64
source_file_path = 'data/source'
target_file_path = 'data/target'

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files
def read_file(file_path):
    file_list = file_name(file_path)
    rank_matrix = []
    for file in file_list:
        df = pd.read_csv(file_path+'/'+file)
        df = df.drop('user_id', axis=1)
        user_rank = np.array(df)
        user_rank_list = user_rank#.tolist()
        rank_matrix.append(user_rank_list)
    return rank_matrix

class MyDataset(Dataset):
    def __init__(self, path):
        #TODO 这里需要修改，因为不知道MF后需要以什么形式传入
        s_u_path = path + '/' + 'source'
        t_u_path = path + '/' + 'target'
        t_i_path = path + '/' + 'target'
        s_u = read_file(s_u_path)
        t_u = read_file(t_u_path)
        t_i = read_file(t_i_path)
        #时间间隔，每三天进行一次训练
        self.time_span = 3
        self.s_u = np.swapaxes(s_u, 0, 1)#.tolist()
        self.t_u = np.swapaxes(t_u, 0, 1)#.tolist()
        self.t_i = np.swapaxes(t_i, 0, 1)#.tolist()
    def __getitem__(self, index):
        user_num =len(self.s_u)
        #用户按照batch_size分，时间也要按照life_span分
        #i表示用户id,j表示第几个周期
        i = index % user_num
        j = index // user_num
        s_u = self.s_u[i,j*self.time_span:(j+1)*self.time_span,:]
        t_u = self.t_u[i,(j+1)*self.time_span,:]
        t_i = self.t_i[i,(j+1)*self.time_span,:]
        return s_u,t_u,t_i

    def __len__(self):
        #根据time_span划分的周期数量
        T_num = (len(self.t_u[0])-1) // self.time_span
        return len(self.s_u)*T_num

if __name__ == '__main__':
    dataset = MyDataset('data')
    dataloader = DataLoader(dataset,Batch_size,shuffle=True)
    for (a,b,c) in dataloader:
        print(a)
