import os
import sys

from util import set_seed
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup


# from model.train import MyDataSet
from model import Encoder,Decoder,Transformer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import config

seed = 42
cuda_index = 0
epoch_size = 20
filename1 = ""
filename2 = ""
target_dir = ""
patience = 3
warmup_steps = 1

# class MyDataSet(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.sourceUTF = torch.rand([80, 4, config.d_model], requires_grad=False)
#         self.targetUF = torch.rand([80, config.d_model], requires_grad=False)
#         self.src, self.trg = [], []
#         for i in range(80):
#             self.src.append(self.sourceUTF[i])
#             self.trg.append(self.targetUF[i])
#
#     def __getitem__(self, index):
#         return self.src[index], self.trg[index]
#
#     def __len__(self):
#         return len(self.src)


class Pipeline:
    def __init__(self):
        set_seed(seed)
        #device = torch.device('cuda:{}'.format(cuda_index) if torch.cuda.is_available() else 'cpu')

    def train_iter(self):
        train_data = tqdm(self.train_data_loader,total = self.train_data_loader.__len__(),file = sys.stdout)

        # 开始数据迭代
        for i_batch,batch_data in enumerate(train_data):
            print(i_batch)
            pred_target = self.transformer(batch_data[0])
            # 暂时还需要调整一下输出的维数
            loss = self.criterion(pred_target,batch_data[1])
            print("Epoch {}, loss:{:.4f}".format(self.global_epcoh, loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()


    def evaluate_iter(self):
        pass

    def run(self):
        best_score = 0
        best_iter = 0
        time_stamp = 0
        # 开始训练
        for epoch in range(epoch_size):
            self.global_epoch = epoch
            # 训练
            self.train_iter()
            # 评估
            score = self.evaluate_iter()
            # 记录最高分数和epoch
            if score > best_score:
                best_score = score
                best_iter = epoch
                torch.save({'epoch':epoch,
                            'best_score':best_score},
                            os.path.join(target_dir,"best_{}".format(epoch)))
            elif epoch - best_iter > patience:
                print("Not upgrade for {} steps, early stopping...".format(patience))
                break

    def forward(self):
        # trainDataSet = MyDataSet()
        # testDataSet = MyDataSet()
        #
        # self.trainLoader = DataLoader(trainDataSet,8,shuffle=True)
        # self.testLoader= DataLoader(testDataSet, 8, shuffle=True)

        # 8个user、4个timestamp、d_model个feature，6个item
        sourceUTF = torch.rand([8, 4, config.d_model], requires_grad=False).to(config.device)
        targetUF = torch.rand([8, config.d_model], requires_grad=False).to(config.device)
        targetIF = torch.rand([config.d_model,6], requires_grad=False).to(config.device)

        data = TensorDataset(sourceUTF, targetUF)
        self.train_data_loader = DataLoader(data, batch_size=5, shuffle=False)

        # 定义transformer
        self.transformer = Transformer().to(config.device)

        # 定义优化器、损失函数等
        self.optimizer = torch.optim.SGD(self.transformer.parameters(), lr=0.001)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
        #                                 num_training_steps= epoch_size * self.trainLoader.__len__())
        self.criterion = torch.nn.MSELoss()

        self.run()


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.forward()