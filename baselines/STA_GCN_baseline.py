import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
from utils.graph_attention_block import GAT_Block


class TemporalUnit(nn.Module):
    def __init__(self, t_kernel, stride, in_channel):
        super(TemporalUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.active = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=(t_kernel, 1),
            stride=(stride, 1),
            padding=(t_kernel // 2, 0)  # pad the time domain
        )
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.dropout = nn.Dropout(0.5, inplace=True)

    def forward(self, x):
        """
        :param x:  x is actually the features after normal GCN
                   it's in shape [batch, channel, frames, joints]

                   therefore, the main job of this unit is to perform convolution
                   on time domain.
        :return:
        """
        b1 = self.active(self.bn1(x))
        b2 = self.conv(b1)
        out = self.dropout(self.bn2(b2))
        return out


class GCNUnit(nn.Module):
    def __init__(self, out_channel, kernel_size, in_channel=3):
        """
        :param out_channel:
                for each adjacent matrix, we have corresponding feature maps with out_channels channel
        :param kernel_size:
                actually it's the num of Adjacent Matrices.
                The original paper use graph partition technique and have various adjacent matrices
        :param in_channel:
                default is 3, because we only have 3D position information at the very first time
        """
        super(GCNUnit, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel * kernel_size,
            kernel_size=(1, 1),
            stride=1
        )
        self.dropout = nn.Dropout(0.5, inplace=True)

    def forward(self, x, adjacency):
        """
        :param x:
                input features in shape [batch, channel, frames, joints]
        :param adjacent:
                adjacent matrices
        :return:
        """
        x1 = self.dropout(self.conv(x))
        b, c, w, h = x1.shape
        x1 = x1.view(b, self.kernel_size, c // self.kernel_size, w, h)

        out = torch.einsum("bncfj, njh -> bcfh", (x1, adjacency))
        return out


class ST_GCN_Block(nn.Module):
    def __init__(self, t_kernel, s_kernel, stride, in_channel, out_channel, residual=True):
        """
        :param t_kernel:        temporal kernel used in temporal convolution unit
        :param s_kernel:        spatial kernel which is same as num of adjacent matrices
        :param stride:
        :param in_channel:

        an ST-GCN block is consisted of a TemporalUnit, GCNUnit and a residual link
        """
        super(ST_GCN_Block, self).__init__()
        self.gcn = GCNUnit(out_channel, s_kernel, in_channel)
        self.tcn = TemporalUnit(t_kernel, stride, out_channel)
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

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adjacent):
        x1 = self.gcn(x, adjacent)
        x2 = self.residual(x)
        out = self.relu(self.tcn(x1) + x2)
        # out = self.relu(x1 + x2)
        return out


class ST_GAT_Block(nn.Module):
    def __init__(self, t_kernel, stride, in_channels, hidden_dim, head=1):
        """
        :param t_kernel:       currently it's still the conv kernel size of temporal unit
        :param stride:
        :param in_channels:
        :param hidden_dim:     graph attention hidden dim which is also going to be the output dim
        :param head:
        """
        super(ST_GAT_Block, self).__init__()
        self.gat = GAT_Block(n_heads=head, in_channels=in_channels, hidden_dim=hidden_dim)
        """
        initially I can't find a effective way to design a temporal graph attention mechanism
        thus, original temporal convolution is used
        """
        self.tgat = TemporalUnit(t_kernel, stride, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adjacent):
        x1 = self.gat(x, adjacent)
        out = self.relu(self.tgat(x1))

        return out


class mlp_head(nn.Module):
    def __init__(self, in_channels, num_class):
        super(mlp_head, self).__init__()
        self.fcn = nn.Conv2d(in_channels=in_channels, out_channels=num_class, kernel_size=1)

    def forward(self, x):
        """
        :param x:
                shared features from ST-GCN backbone after global max-pooling, in shape [batch, in_channels, 1, 1]
        :return:
        """
        x = self.fcn(x)
        out = x.squeeze()
        # keep "batch = 1" shape
        if len(out.shape) == 1:
            out = torch.unsqueeze(out, dim=0)

        return out


class lstm_head(nn.Module):
    def __init__(self, in_channels, num_class, hidden, joints=23):
        super(lstm_head, self).__init__()
        # LSTM Classification Head
        self.rnn = nn.LSTM(batch_first=True,
                           num_layers=1,
                           hidden_size=hidden,
                           input_size=joints * in_channels)  # joints_num * channels
        self.fc = nn.Linear(hidden, num_class)

    def forward(self, x):
        """
        :param x:    same as mlp_head
        :return:
        """
        out, _ = self.rnn(x)
        x = out[:, -1, :]  # take the output of the last time step
        out = self.fc(x)

        return out


class STA_GCN(nn.Module):
    def __init__(self, in_channels, num_class, joints=22, edge_importance_weighting=True, max_hop=1,
                 classify_head="lstm"):
        """
        :param in_channels:
        :param num_class:
                            number of category
        :param joints:
                            human body joints used
        :param edge_importance_weighting:
        :param max_hop:
        :param classify_head:
                            if dataset is padded, then rnn-like classification head can be used
                            otherwise, each gait cycle takes occupied different time
        :param hidden:      hidden size of rnn cell
        """
        super(STA_GCN, self).__init__()
        self.head = classify_head
        self.graph = Graph(max_hop)
        # the adjacency matrix does not need to update
        adjacency = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # adjacency matrix isn't a model parameter and if we don't register, the model state_dict doesn't contain it
        # all in all, avoid updating while saving it in dict
        self.register_buffer("adjacency", adjacency)
        t_kernel = 9
        s_kernel = adjacency.shape[0]
        self.data_bn = nn.BatchNorm1d(in_channels * adjacency.shape[1])
        self.st_gcn = nn.ModuleList((
            # ST GCN Tiny
            ST_GCN_Block(t_kernel, s_kernel, 1, in_channels, 64, residual=False),
            ST_GAT_Block(t_kernel, s_kernel, in_channels=64, hidden_dim=64),
            ST_GCN_Block(t_kernel, s_kernel, 2, 64, 128, residual=True),
            ST_GAT_Block(t_kernel, s_kernel, in_channels=128, hidden_dim=128),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(adjacency))
                for i in self.st_gcn
            ])

        embed_dim = 128

        if self.head == "lstm":
            self.classify = lstm_head(in_channels=embed_dim, num_class=num_class, hidden=128, joints=joints)
        else:
            self.classify = mlp_head(in_channels=embed_dim, num_class=num_class)

    def forward(self, x):
        batch, channel, frames, joints = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch, channel * joints, frames)
        x = self.data_bn(x)
        x = x.view(batch, channel, joints, frames)
        x = x.permute(0, 1, 3, 2).contiguous()

        # forward
        for gcn, importance in zip(self.st_gcn, self.edge_importance):
            x = gcn(x, self.adjacency * importance)

        # LSTM Classification Head
        if self.head == "lstm":
            x = x.permute(0, 2, 3, 1).flatten(2)
        else:
            # global pooling
            x = F.avg_pool2d(x, x.size()[2:])

        out = self.classify(x)

        return out

# code for debugging
# x = torch.randn(3, 3, 390, 19)
# net = ST_GCN(x.shape[1], 10)
# print(net(x).shape)
