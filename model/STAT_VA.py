import torch
import torch.nn as nn
import torch.nn.functional as F


class STAT_VA(nn.Module):
    def __init__(self, in_channels, joints, hidden=None):
        super(STAT_VA, self).__init__()

        # self.q = nn.Conv2d(in_channels=in_channels, out_channels=hidden, kernel_size=1)
        # self.k = nn.Conv2d(in_channels=in_channels, out_channels=hidden, kernel_size=1)
        # self.v = nn.Conv2d(in_channels=in_channels, out_channels=hidden, kernel_size=1)

        self.vattn = nn.Parameter(torch.ones((1, 1, 1, joints)), requires_grad=True)
        self.channel_attention = CA_Blcok(in_channels=in_channels)

        # self.temp = hidden ** (-1 / 2)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, stat, features):
        """
        :param stat:            stat(unaffected) - stat(affected)      [B, C, 8, 4]
        :param features:        [B, C, 8, 4]   <-  attention
        :return:
        """
        b, c, f, j = features.shape

        # mask = torch.norm(input=stat, p=1, dim=(0, 1, 2)).unsqueeze(0)  # [B, 4]
        mask = torch.norm(input=stat, p=1, dim=(1, 2))
        mask = F.normalize(mask, dim=1, p=1)

        '''[B, hidden, 8, 4]'''
        # query = torch.permute(self.q(features), dims=[0, 2, 3, 1]).contiguous()   # [B, frames, 4, hidden]
        # key = torch.permute(self.k(features), dims=[0, 2, 1, 3]).contiguous()     # [B, frames, hidden, 4]

        # query = features.permute(0, 2, 3, 1).contiguous()
        # key = features.permute(0, 2, 1, 3).contiguous()

        # attn = torch.matmul(query, key)  # [B, frames, 4, 4]
        # attn = F.softmax(attn * self.temp, dim=-1)
        # attn = self.dropout(self.vattn)

        self.vattn = self.vattn.to(features.device)  # learnable adjacency (affinity) matrix [1, 1, 1, 4]

        mask = mask[:, None, None, :]
        attn = torch.add(self.vattn, mask)  # [B, 1, 1, 4]
        attn = F.softmax(attn, dim=-1)

        # value = self.v(features)
        # value = self.v(features).permute(0, 2, 3, 1)                      # [B, hidden, 8, 4]
        # value = features.permute(0, 2, 3, 1).contiguous()                 # [B, 8, 4, in_channels]

        value = self.channel_attention(features)
        out = torch.multiply(value, attn)

        return out


class CA_Blcok(nn.Module):
    # Channel Wise Attention
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super(CA_Blcok, self).__init__(*args, **kwargs)
        self.attn = nn.Parameter(torch.ones((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        """
            x : [B, C, 8, J]
        """
        self.attn = self.attn.to(x.device)
        out = torch.multiply(x, self.attn)
        return out


