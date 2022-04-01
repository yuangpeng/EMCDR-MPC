import csv
import pandas as pd
import numpy as np
import os
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
        user_rank_list = user_rank.tolist()
        rank_matrix.append(user_rank_list)
    return rank_matrix



if __name__ == '__main__':
    context = read_file(source_file_path)
    # print(context)
    context = read_file(target_file_path)
    # print(context)