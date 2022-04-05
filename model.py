import torch
import numpy as np
import pandas as pd
import config
import util
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.dk = config.d_model // config.multi_heads
        self.device = config.device
        self.W_Qs = torch.nn.ModuleList(torch.nn.Linear(config.d_model, self.dk, bias=False)
                                        for _ in range(config.multi_heads * config.num_layers_encoder))
        self.W_Ks = torch.nn.ModuleList(torch.nn.Linear(config.d_model, self.dk, bias=False)
                                        for _ in range(config.multi_heads * config.num_layers_encoder))
        self.W_Vs = torch.nn.ModuleList(torch.nn.Linear(config.d_model, self.dk, bias=False)
                                        for _ in range(config.multi_heads * config.num_layers_encoder))

        for head in range(config.multi_heads * config.num_layers_encoder):
            util.init_linear_wt(self.W_Qs[head])
            util.init_linear_wt(self.W_Ks[head])
            util.init_linear_wt(self.W_Vs[head])

        self.W0 = torch.nn.ModuleList(torch.nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(config.num_layers_encoder))
        for layer in range(config.num_layers_encoder):
            util.init_linear_wt(self.W0[layer])

        self.W_residua = torch.nn.ModuleList(torch.nn.Linear(config.d_model, config.d_model) for _ in range(config.num_layers_encoder))
        for layer in range(config.num_layers_encoder):
            util.init_linear_wt(self.W_residua[layer])

        self.W_forward0 = torch.nn.ModuleList(torch.nn.Linear(config.d_model, config.d_model) for _ in range(config.num_layers_encoder))
        self.W_forward1 = torch.nn.ModuleList(torch.nn.Linear(config.d_model, config.d_model) for _ in range(config.num_layers_encoder))
        for layer in range(config.num_layers_encoder):
            util.init_linear_wt(self.W_forward0[layer])
            util.init_linear_wt(self.W_forward1[layer])

    def forward(self, X):
        # X.shape = [batch_size, seq_len, embedding_size]; embed_size = d_model
        batch_size, seq_len, embed_size = X.size()

        LN = torch.nn.LayerNorm([seq_len, config.d_model]).to(self.device) # 忽略batch的维度

        """ 位置编码 """
        pos_enc = np.array([[pos / np.power(1000, 2 * (i // 2) / config.d_model) for i in range(config.d_model)]
                            for pos in range(seq_len)], dtype=np.float32)
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pos_enc = torch.from_numpy(pos_enc).to(self.device)  # pos_enc.shape = [seq_len, d_model]

        en_input = X + pos_enc  # [batch_size, seq_len, embed_size]
        # en_mask = en_mask.unsqueeze(1).expand(-1, seq_len, -1) # pad_mask
        dk = torch.sqrt(torch.tensor(self.dk))
        for layer in range(config.num_layers_encoder):
            """ 多头注意力机制 """
            multi_attentions = None
            for head in range(config.multi_heads):
                Q = self.W_Qs[layer * config.multi_heads + head](en_input)  # [batch_size, seq_len, d_model // multi_heads]
                K = self.W_Ks[layer * config.multi_heads + head](en_input)
                V = self.W_Vs[layer * config.multi_heads + head](en_input)
                weight = torch.bmm(Q, K.transpose(1, 2)) / dk  # [batch_size, seq_len, seq_len]
                # weight = weight.masked_fill(en_mask, value=-float('inf'))
                dot_attention = torch.bmm(torch.softmax(weight, dim=2), V)
                if multi_attentions == None:
                    multi_attentions = dot_attention  # [batch_size, seq_len, d_model // multi_heads]
                else:
                    multi_attentions = torch.cat([multi_attentions, dot_attention], dim=2)

            multi_attentions = self.W0[layer](multi_attentions)  # [batch_size, seq_len, d_model]

            """ 残差与归一化 """
            residua_outcome0 = en_input + multi_attentions  # [batch_size, seq_len, d_model]
            residua_outcome0 = LN(residua_outcome0) # layer normalization
            residua_outcome0 = self.W_residua[layer](residua_outcome0)

            """ 前馈传播 """
            forward_outcome = self.W_forward1[layer](F.relu(self.W_forward0[layer](residua_outcome0)))

            """ 残差与归一化 """
            residua_outcome1 = forward_outcome + residua_outcome0
            residua_outcome1 = LN(residua_outcome1)
            en_input = self.W_residua[layer](residua_outcome1)

        return en_input

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dk = config.d_model // config.multi_heads
        self.device = config.device
        self.W_Qs = torch.nn.ModuleList(torch.nn.Linear(config.d_model, self.dk, bias=False)
                                          for _ in range(config.multi_heads * config.num_layers_decoder))
        for head in range(config.multi_heads * config.num_layers_decoder):
            util.init_linear_wt(self.W_Qs[head])

        self.W0 = torch.nn.ModuleList(
            torch.nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(config.num_layers_decoder))
        for layer in range(config.num_layers_encoder):
            util.init_linear_wt(self.W0[layer])

        self.W_residua = torch.nn.ModuleList(
            torch.nn.Linear(config.d_model, config.d_model) for _ in range(config.num_layers_decoder))
        for layer in range(config.num_layers_encoder):
            util.init_linear_wt(self.W_residua[layer])

        self.W_forward0 = torch.nn.ModuleList(
            torch.nn.Linear(config.d_model, config.d_model) for _ in range(config.num_layers_decoder))
        self.W_forward1 = torch.nn.ModuleList(
            torch.nn.Linear(config.d_model, config.d_model) for _ in range(config.num_layers_decoder))
        for layer in range(config.num_layers_encoder):
            util.init_linear_wt(self.W_forward0[layer])
            util.init_linear_wt(self.W_forward1[layer])

        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, config.d_model, (k, config.d_model)) for k in config.filter_sizes])
        for cnn in self.convs:
            util.init_cnn_wt(cnn)

        self.dropout = torch.nn.Dropout(config.dropout)

        self.linear = torch.nn.Linear(len(config.filter_sizes) * config.d_model, config.d_model)
        util.init_linear_wt(self.linear)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)) # [batch_size, num_filters, seq_len-k+1, 1]
        x = x.squeeze(3) # [batch_size, num_filters, seq_len-k+1]
        x = F.max_pool1d(x, kernel_size=x.size(2)) # [batch_size, num_filters, 1]
        x = x.squeeze(2) # [batch_size, num_filters]
        return x

    def forward(self, X, W_Ks, W_Vs):
        # X.shape = [batch_size, seq_len, config.d_model]
        batch_size, seq_len, d_model = X.size()

        LN = torch.nn.LayerNorm([seq_len, d_model]).to(self.device)

        dk = torch.sqrt(torch.tensor(self.dk))
        for layer in range(config.num_layers_decoder):
            multi_attentions = None
            for head in range(config.multi_heads):
                Q = self.W_Qs[layer * config.multi_heads + head](X)
                K = W_Ks[head](X) # encoder最后一层
                V = W_Vs[head](X) # encoder最后一层
                weight = torch.bmm(Q, K.transpose(1, 2)) / dk
                dot_attention = torch.bmm(torch.softmax(weight, dim=2), V)
                if multi_attentions == None:
                    multi_attentions = dot_attention  # [batch_size, seq_len, d_model // multi_heads]
                else:
                    multi_attentions = torch.cat([multi_attentions, dot_attention], dim=2)

            multi_attentions = self.W0[layer](multi_attentions)  # [batch_size, seq_len, d_model]

            """ 残差与归一化 """
            residua_outcome0 = X + multi_attentions  # [batch_size, seq_len, d_model]
            residua_outcome0 = LN(residua_outcome0)
            residua_outcome0 = self.W_residua[layer](residua_outcome0)

            """ 前馈传播 """
            forward_outcome = self.W_forward1[layer](F.relu(self.W_forward0[layer](residua_outcome0)))

            """ 残差与归一化 """
            residua_outcome1 = forward_outcome + residua_outcome0
            residua_outcome1 = LN(residua_outcome1)
            X = self.W_residua[layer](residua_outcome1)

        # X.shape = [batch_size, seq_len, d_model]
        X = torch.unsqueeze(X, dim=1)
        X = torch.cat([self.conv_and_pool(X, conv) for conv in self.convs], 1) # [batch_size, 2 * d_model]

        X = self.dropout(X)

        X = self.linear(X)

        return X

test_date = torch.rand([8, 4, config.d_model], requires_grad=False).to(config.device)
encoder = Encoder().to(config.device)
X = encoder(test_date)
print(X.size())
decoder = Decoder().to(config.device)
X = decoder(X, encoder.W_Ks[config.multi_heads * (config.num_layers_encoder - 1):], encoder.W_Vs[config.multi_heads * (config.num_layers_encoder - 1):])
print(X.size())