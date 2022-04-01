import csv
import os
file_path = 'data'

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

def read_file(file_path):
    file_list = file_name(file_path)
    content = []
    for file in file_list:
        print(file)
        with open(file_path+'/'+file, 'r',encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                content+=row
    while '' in content:
        content.remove('')
    return content

if __name__ == '__main__':
    file_list = file_name(file_path)
    print(file_list)
