import math

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, einsum
from torch.autograd import Variable
from torch import linalg as LA
from einops import rearrange, repeat

from graph import Graph


def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu'):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                activation_factory(activation),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            activation_factory(activation),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.act = activation_factory(activation)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        return out


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(UnitTCN, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.scale = hidden_dim ** -0.5
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim * 2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x):
        y = rearrange(x, 'n c t v -> n t v c').contiguous()
        y = self.ln(y)
        y = self.to_qk(y)
        qk = y.chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b t v (h d) -> (b t) h v d', h=self.n_heads), qk)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1).float()
        return attn


class SA_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SA_GC, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head = A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

        rel_channels = in_channels // 8
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)

    def forward(self, x, attn=None):
        N, C, T, V = x.size()

        out = None
        if attn is None:
            attn = self.attn(x)
        A = attn * self.shared_topology.unsqueeze(0)  # 用natural link refine 推理得到的 SA adjacency
        for h in range(self.num_head):
            A_h = A[:, h, :, :]  # (nt)vv
            feature = rearrange(x, 'n c t v -> (n t) v c')  # 所有 frames share 同一个 SA refined adjacency
            z = A_h @ feature
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)

        return out


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(EncodingBlock, self).__init__()
        self.agcn = SA_GC(in_channels, out_channels, A)
        self.tcn = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=5, stride=stride,
                                           dilations=[1, 2], residual=False)

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, attn=None):
        y = self.relu(self.tcn(self.agcn(x, attn)) + self.residual(x))
        return y


class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, graph=None, in_channels=3,
                 drop_out=0, num_head=3, noise_ratio=0.1, k=0, gain=1, max_hop=1):
        super(InfoGCN, self).__init__()

        self.graph = Graph(max_hop)

        # A = np.stack([np.eye(num_point)] * num_head, axis=0)
        A = self.graph.A

        base_channel = 64
        self.num_class = num_class[0]
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(base_channel * num_point)
        self.noise_ratio = noise_ratio
        self.z_prior = torch.empty(num_class[0], base_channel * 2)

        self.A_vector = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        self.gain = gain
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))

        self.l1 = EncodingBlock(base_channel, base_channel, A)
        self.l4 = EncodingBlock(base_channel, base_channel * 2, A, stride=2)
        self.l5 = EncodingBlock(base_channel * 2, base_channel * 2, A)

        self.fc = nn.Linear(base_channel * 2, base_channel * 2)
        self.fc_mu = nn.Linear(base_channel * 2, base_channel * 2)
        self.fc_logvar = nn.Linear(base_channel * 2, base_channel * 2)
        self.decoder = nn.Linear(base_channel * 2, num_class[0])

        nn.init.orthogonal_(self.z_prior, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / num_class[0]))
        bn_init(self.data_bn, 1)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x):
        N, C, T, V = x.size()
        x = rearrange(x, 'n c t v -> (n t) v c', v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N * T, -1, -1) @ x

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n t) v c -> n (v c) t', t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (v c) t -> n c t v', v=V).contiguous()

        x = self.l1(x)
        x = self.l4(x)
        x = self.l5(x)

        # N,C,T,V
        c_new = x.size(1)
        x = x.view(N, c_new, -1)
        x = x.mean(2)

        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        return y_hat, z


def get_mmd_loss(z, z_prior, y, num_cls):
    """
    :param z:               [batch, 2 * base_channels]
    :param z_prior:
    :param y:
    :param num_cls:
    :return:
    """
    # 当前batch中存在的label打上标签
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    # 每一个class隐空间的mu     [num_class, 2 * base_channels]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    # 去逼近 mean 0 ？？
    l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
    # 每一个class的mean去逼近先验z， 先验z把label的隐空间表达互相整合成2*base_channel空间下正交的向量
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]