import os
import sys

from utils import set_seed
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup


from model.train import MyDataSet
from model import Encoder,Decoder
import config

seed = 42
cuda_index = 0
epoch_size = 20
filename1 = ""
filename2 = ""
target_dir = ""
patience = 3
warmup_steps = 1


class Pipeline:
    def __init__(self, args):
        set_seed(seed)
        device = torch.device('cuda:{}'.format(cuda_index) if torch.cuda.is_available() else 'cpu')

    def train_iter(self):
        train_data = tqdm(self.trainLoader,total = self.trainLoader.__len__(),file = sys.stdout)
        count, loss = 0, 0
        # 开始数据迭代
        for sourceUTF,targetUF in enumerate(train_data):
            pred_target = self.encoder(sourceUTF)
            pred_target = self.decoder(pred_target, self.encoder.W_Ks[config.multi_heads * (config.num_layers_encoder - 1):], self.encoder.W_Vs[config.multi_heads * (config.num_layers_encoder - 1):])
            loss = self.criterion(pred_target,targetUF)
            self.optimizer.zero_grad()
            loss.backward()
            self.scheduler.step()


    def evaluate_iter(self):
        pass

    def run(self):
        best_score = 0
        best_iter = 0
        time_stamp = 0
        # 开始训练
        for epoch in range(epoch_size):
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
        trainDataSet = MyDataSet(filename1)
        testDataSet = MyDataSet(filename1)

        self.trainLoader = DataLoader(trainDataSet,16,shuffle=True)
        self.testLoader= DataLoader(testDataSet, 16, shuffle=True)

        params = []
        # 定义优化器、损失函数等
        self.optimizer = torch.optim.SGD(params, lr=0.1)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                        num_training_steps= epoch_size * self.trainLoader.__len__())
        self.criterion = F.mse_loss()

        self.encoder = Encoder().to(config.device)
        self.decoder = Decoder().to(config.device)


        self.run()