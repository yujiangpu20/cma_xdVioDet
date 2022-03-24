import torch

from layers import *
import torch.nn.functional as F
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)


class CMA_LA(nn.Module):
    def __init__(self, modal_a, modal_b, hid_dim=128, d_ff=512, dropout_rate=0.1):
        super(CMA_LA, self).__init__()

        self.cross_attention = CrossAttention(modal_b, modal_a, hid_dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(modal_a, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(d_ff, 128, kernel_size=1),
            nn.Dropout(dropout_rate),
        )
        self.norm = nn.LayerNorm(modal_a)

    def forward(self, x, y, adj):
        new_x = x + self.cross_attention(y, x, adj)
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 2, 1)
        new_x = self.ffn(new_x)

        return new_x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_features = args.feature_size
        n_class = args.num_classes
        self.dis_adj = DistanceAdj()

        self.cross_attention = CMA_LA(modal_a=1024, modal_b=128, hid_dim=128, d_ff=512)
        self.classifier = nn.Conv1d(128, 1, 7, padding=0)
        self.apply(weight_init)

    def forward(self, x):
        f_v = x[:, :, :1024]
        f_a = x[:, :, 1024:]
        adj = self.dis_adj(f_v.shape[0], f_v.shape[1])

        new_v = self.cross_attention(f_v, f_a, adj)
        new_v = F.pad(new_v, (6, 0))
        logits = self.classifier(new_v)
        logits = logits.squeeze(dim=1)
        logits = torch.sigmoid(logits)

        return logits
