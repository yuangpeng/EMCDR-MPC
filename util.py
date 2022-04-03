import config
import torch

def init_lstm_wt(lstm):
    for name, _ in lstm.named_parameters():
        if 'weight' in name:
            wt = getattr(lstm, name)
            # 在-0.2至0.2之间生成一个随机数
            torch.nn.init.xavier_normal_(wt)
        elif 'bias' in name:
            # set forget bias to 1
            bias = getattr(lstm, name)
            n = bias.size()[0]
            start, end = n // 4, n // 2
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    # 权重正态化
    torch.nn.init.xavier_normal_(linear.weight)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    torch.nn.init.xavier_normal_(wt)

