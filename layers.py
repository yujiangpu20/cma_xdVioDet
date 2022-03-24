import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
import torch.nn.init as torch_init
import numpy as np
import math


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=2500):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model1, d_model2, dim_k, n_heads=1):
        super(CrossAttention, self).__init__()
        self.d_model1 = d_model1
        self.d_model2 = d_model2
        self.dim_k =dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model1, dim_k)
        self.k = nn.Linear(d_model2, dim_k)
        self.v = nn.Linear(d_model2, dim_k)

        self.o = nn.Linear(dim_k, d_model2)
        self.norm_fact = 1 / math.sqrt(dim_k)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x, y, adj):
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)

        attention_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        attention_map = self.act(attention_map + adj)
        temp = torch.matmul(attention_map, V).reshape(y.shape[0], y.shape[1], -1)
        output = self.o(temp).reshape(-1, y.shape[1], y.shape[2])

        return output


class DistanceAdj(nn.Module):
    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.bias = nn.Parameter(torch.FloatTensor(1))

    def forward(self, batch_size, max_seqlen):
        self.arith = np.arange(max_seqlen).reshape(-1, 1)
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
        self.dist = torch.from_numpy(squareform(dist)).cuda()
        self.dist = torch.exp(- torch.abs(self.w * (self.dist**2) + self.bias))
        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1).cuda()

        return self.dist