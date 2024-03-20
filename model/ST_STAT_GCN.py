import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
from model.STAT_net import *


class TemporalUnit(nn.Module):
    def __init__(self, t_kernel, stride, in_channel, groups):
        super(TemporalUnit, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_channel)
        # self.active = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=(t_kernel, 1),
            stride=(stride, 1),
            groups=groups,
            padding=(t_kernel // 2, 0),  # pad the time domain
            # padding_mode='replicate',
            bias=False
        )

    def forward(self, x):
        """
        :param x:  x is actually the features after normal GCN
                   it's in shape [batch, channel, frames, joints]

                   therefore, the main job of this unit is to perform convolution
                   on time domain.
        :return:
        """
        # b1 = self.active(self.bn1(x))
        # b2 = self.conv(b1)
        # out = self.dropout(self.bn2(b2))
        # out = self.bn2(b2)

        out = self.conv(x)

        return out


class GCNUnit(nn.Module):
    def __init__(self, out_channel, groups, kernel_size=1, in_channel=3):
        """
        :param out_channel:
                for each adjacent matrix, we have corresponding feature maps with out_channels channel
        :param groups:
                for each modal, we use unique filters to get more meaningful features.
        :param kernel_size:
                actually it's the num of Adjacent Matrices.
                The original paper use graph partition technique and have various adjacent matrices
        :param in_channel:
                default is 3, because we only have 3D position information at the very first time
        """
        super(GCNUnit, self).__init__()
        assert out_channel % groups == 0 and in_channel % groups == 0, "conv set error"
        self.kernel_size = kernel_size
        """
        use group convolution instead to find different patterns
        of different features (position, velocity, acceleration etc.)
        """
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel * kernel_size,
            kernel_size=(1, 1),
            stride=1,
            groups=groups
        )
        self.dropout = nn.Dropout(0.5, inplace=True)

    def forward(self, x, adjacent):
        """
        :param x:
                input features in shape [batch, channel, frames, joints]
        :param adjacent:
                adjacent matrices
        :return:
        """
        x1 = self.dropout(self.conv(x))
        b, c, w, h = x1.shape

        out = torch.einsum("bcfj, jh -> bcfh", (x1, adjacent))
        return out


class ST_GCN_Block(nn.Module):
    def __init__(self, t_kernel, s_kernel, stride, in_channel, out_channel, groups,
                 ln=False, insn=False, residual=True):
        """
        :param t_kernel:        temporal kernel used in temporal convolution unit
        :param s_kernel:        spatial kernel which is same as num of adjacent matrices
        :param stride:
        :param in_channel:

        an ST-GCN block is consisted of a TemporalUnit, GCNUnit and a residual link
        """
        super(ST_GCN_Block, self).__init__()
        self.gcn = GCNUnit(out_channel, groups, s_kernel, in_channel)
        self.tcn = TemporalUnit(t_kernel, stride, out_channel, groups)

        self.layernorm = ln
        self.instancenorm = insn

        self.LN = nn.LayerNorm(out_channel)
        self.IN = nn.InstanceNorm2d(num_features=out_channel)

        if not residual:
            self.residual = lambda x: 0
        elif out_channel == in_channel and stride == 1:
            # we will automatically do padding in tcn to fit the temporal kernel
            self.residual = nn.Identity()
        else:
            # for tcn, the time axis size formula is
            # (frames + 2 * (t_kernel // 2) - t_kernel) / stride + 1
            # we force the t_kernel to be an odd number, then it can be simplified to : (frames - 1) / stride + 1
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    1,
                    (stride, 1),  # (frames - 1) / stride + 1
                ),
                nn.BatchNorm2d(out_channel)
            )

        self.relu = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x, adjacent):
        x1 = self.gcn(x, adjacent)

        x2 = self.residual(x)

        if self.layernorm:
            x3 = self.tcn(x1)
            x3 = x3.permute(0, 2, 3, 1).contiguous()
            x3 = self.LN(x3)
            x3 = x3.permute(0, 3, 1, 2).contiguous()
            out = self.relu(x3 + x2)
        else:
            out = self.relu(self.tcn(x1) + x2)

        return out


class ST_STAT_GCN(nn.Module):
    def __init__(self, in_channels, num_class, groups, affected_idx: list, unaffected_idx: list,
                 item, channels: list, stat_embed, max_hop=1, version='V2', attn=True):
        """
        :param in_channels:
        :param num_class:
                            a list of items classes
        :param joints:
                            human body joints used
        :param edge_importance_weighting:
        :param max_hop:
        """
        super(ST_STAT_GCN, self).__init__()
        self.graph = Graph(max_hop)
        # the adjacency matrix does not need to update
        adjacency = torch.tensor(self.graph.adjacency, dtype=torch.float32, requires_grad=False)
        # adjacency matrix isn't a model parameter and if we don't register, the model state_dict doesn't contain it
        # all in all, avoid updating while saving it in dict
        self.register_buffer("adjacency", adjacency)

        self.gnorm = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self.insnorm = nn.InstanceNorm2d(num_features=in_channels)

        self.item = item

        t_kernel = 3

        self.l1 = ST_GCN_Block(t_kernel, 1, 1, in_channels, channels[0], groups,
                               ln=False, insn=False, residual=True)

        self.l2 = ST_GCN_Block(t_kernel, 1, 1, channels[0], channels[1], groups,
                               ln=False, insn=False, residual=False)

        self.l3 = ST_GCN_Block(t_kernel, 1, 1, channels[1], channels[1], groups,
                               ln=False, insn=False, residual=False)

        if version == 'V1':
            self.stat_net = STAT_Net(
                affect_idx=affected_idx,
                unaffected_idx=unaffected_idx,
                in_channels=channels[-1],
                embed_dim=stat_embed,
                classes=num_class,
            )
        elif version == 'V2':
            self.stat_net = STAT_Net_V2(
                affect_idx=affected_idx,
                unaffected_idx=unaffected_idx,
                in_channels=channels[-1],
                embed_dim=stat_embed,
                classes=num_class,
            )
        elif version == 'V3':
            self.stat_net = STAT_Net_V3(
                affect_idx=affected_idx,
                unaffected_idx=unaffected_idx,
                in_channels=channels[-1],
                embed_dim=stat_embed,
                classes=num_class,
            )

        self.stat_attn = attn

    def forward(self, x, label=None, get_cl_loss=False, names=None):

        if self.item == 12:
            x1 = self.insnorm(x)
        else:
            x1 = self.gnorm(x)

        # forward
        f1 = self.l1(x1, self.adjacency)
        f2 = self.l2(f1, self.adjacency)
        f3 = self.l3(f2, self.adjacency)

        if get_cl_loss:
            logits, cl_loss = self.stat_net(f3, label, get_cl_loss, names, self.stat_attn)
            return logits, cl_loss
        else:
            logits = self.stat_net(f3, attn=self.stat_attn)
            return logits
