import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.FR_Head import CL_Module
from model.STAT_VA import STAT_VA


class STAT_Net(nn.Module):
    def __init__(self, affect_idx: list, unaffected_idx: list, in_channels, embed_dim, classes):
        super(STAT_Net, self).__init__()

        self.joints = len(affect_idx)
        self.affect = affect_idx
        self.unaffect = unaffected_idx

        self.num_class = classes

        self.VA_Block = STAT_VA(
            in_channels=in_channels,
            joints=self.joints
        )

        self.fuse1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, self.joints),
            stride=1,
            padding=0
        )

        self.fuse2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, self.joints),
            stride=1,
            padding=0
        )

        self.mlp_classify = nn.Sequential(
            # nn.Linear(in_features=16 * embed_dim, out_features=hidden),
            # nn.Dropout(0.4),
            # nn.LeakyReLU(),
            # nn.Linear(in_features=hidden, out_features=self.num_class)

            nn.Linear(in_features=18 * embed_dim, out_features=self.num_class)
        )

        self.CL_Module = CL_Module(
            channels=embed_dim * 9,
            classes=self.num_class,
            use_p_map=True
        )

    def get_stat_features(self, x):
        """
        :param x:     features in shape [b, c, f, j]
        :return:
                    calculate statistical information for each joint of each channel
                    in this way, we can squeeze temporal information
                    [b, c, 8, j]   '8' stands for 8 basic statistical values
        """

        s1, _ = torch.min(x, dim=-2, keepdim=True)
        s2, _ = torch.max(x, dim=-2, keepdim=True)
        s3 = torch.quantile(x, dim=-2, q=torch.tensor([0.25]).to(x.device), keepdim=True)[0]
        s4, _ = torch.median(x, dim=-2, keepdim=True)
        s5 = torch.quantile(x, dim=-2, q=torch.tensor([0.75]).to(x.device), keepdim=True)[0]
        s6 = torch.mean(x, dim=-2, keepdim=True)
        s7 = torch.std(x, dim=-2, keepdim=True)
        s8 = s2 - s1  # peak-to-peak

        # stat_features = torch.concat([s1, s2, s3, s4, s5, s6, s7, s8], dim=-2)

        diff_x = torch.clone(x.detach()).to(x.device)
        diff_x[:, :, 1:, :] = x[:, :, :-1, :]
        diff = x - diff_x
        s_10 = torch.norm(input=diff, p=1, dim=-2, keepdim=True)  # Total Variatuon

        stat_features = torch.concat([s2, s3, s4, s5, s6, s7, s8, s_10, s1], dim=-2)
        return stat_features

    def forward(self, features, label=None, get_cl_loss=False, names=None, attn=True):
        unaffected_features = features[:, :, :, self.unaffect]
        affected_features = features[:, :, :, self.affect]

        stat_affect = self.get_stat_features(affected_features)  # [B, C, 9, 4]
        stat_unaffect = self.get_stat_features(unaffected_features)  # [B, C, 9, 4]
        mask = stat_unaffect - stat_affect

        if attn:
            unaffected_features = unaffected_features + self.VA_Block(mask, unaffected_features)
            affected_features = affected_features + self.VA_Block(mask, affected_features)

        stat1 = self.get_stat_features(unaffected_features + affected_features)

        stat1 = self.fuse1(stat1)  # [B, C, 9, 1]
        stat1 = torch.flatten(stat1, start_dim=1)
        stat_features = stat1

        stat2 = torch.concat([stat_affect, stat_unaffect], dim=-2)  # [B, C, 18, 4]
        stat2 = self.fuse2(stat2)  # [B, C, 18, 1]
        stat2 = torch.flatten(stat2, start_dim=1)

        logits = self.mlp_classify(stat2)

        if get_cl_loss:
            cl_loss = self.CL_Module(stat_features, label, logits)
            return logits, cl_loss
        else:
            return logits


class STAT_Net_V2(nn.Module):
    def __init__(self, affect_idx: list, unaffected_idx: list, in_channels, embed_dim, classes):
        super(STAT_Net_V2, self).__init__()

        self.joints = len(affect_idx)
        self.affect = affect_idx
        self.unaffect = unaffected_idx

        self.num_class = classes

        self.VA_Block = STAT_VA(
            in_channels=in_channels,
            joints=self.joints
        )

        self.fuse1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, self.joints),
            stride=1,
            padding=0
        )

        self.fuse2 = nn.Conv2d(
            # attn_hid = in_channels since we don't use q^T K V
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, self.joints),
            stride=1,
            padding=0
        )

        self.mlp_classify = nn.Sequential(
            # nn.Linear(in_features=16 * embed_dim, out_features=hidden),
            # nn.Dropout(0.3),
            # nn.ReLU(),
            # nn.Linear(in_features=hidden, out_features=self.num_class)

            nn.Linear(in_features=16 * embed_dim, out_features=self.num_class)
        )

        self.CL_Module = CL_Module(
            channels=embed_dim * 8,
            classes=self.num_class,
            use_p_map=True
        )

    def get_stat_features(self, x, proto=True):
        """
        :param x:     features in shape [b, c, f, j]
        :return:
                    calculate statistical information for each joint of each channel
                    in this way, we can squeeze temporal information
                    [b, c, 8, j]   '8' stands for 8 basic statistical values
        """
        s1, _ = torch.min(x, dim=-2, keepdim=True)
        s2, _ = torch.max(x, dim=-2, keepdim=True)
        s3 = torch.quantile(x, dim=-2, q=torch.tensor([0.25]).to(x.device), keepdim=True)[0]
        s4, _ = torch.median(x, dim=-2, keepdim=True)
        s5 = torch.quantile(x, dim=-2, q=torch.tensor([0.75]).to(x.device), keepdim=True)[0]
        s6 = torch.mean(x, dim=-2, keepdim=True)
        s7 = torch.std(x, dim=-2, keepdim=True)
        s8 = s2 - s1  # peak-to-peak
        s9_V1 = torch.mean(torch.pow((x - s6) / s7, 3), dim=-2, keepdim=True)  # Skewness
        s9_V2 = torch.mean(torch.pow((x - s6) / s7, 4), dim=-2, keepdim=True) - 3  # Kurtosis

        if proto:
            stat_features = torch.concat([s2, s3, s4, s5, s6, s7, s8, s9_V1], dim=-2)
        else:
            stat_features = torch.concat([s2, s3, s4, s5, s6, s7, s8, s9_V2], dim=-2)

        return stat_features

    def forward(self, features, label=None, get_cl_loss=False, names=None, attn=True):
        unaffected_features = features[:, :, :, self.unaffect]
        affected_features = features[:, :, :, self.affect]

        stat_affect = self.get_stat_features(affected_features, proto=False)  # [B, C, 9, 4]
        stat_unaffect = self.get_stat_features(unaffected_features, proto=False)  # [B, C, 9, 4]
        mask = stat_unaffect - stat_affect

        if attn:
            unaffected_features = unaffected_features + self.VA_Block(mask, unaffected_features)
            affected_features = affected_features + self.VA_Block(mask, affected_features)

        stat1 = self.get_stat_features(unaffected_features + affected_features)
        stat1 = self.fuse1(stat1)  # [b, embed, 9, 1]
        stat1 = torch.flatten(stat1, start_dim=1)  # [b, 9 * embed]

        """
        用attention之后的特征取stat values
        """
        # stat_affect = self.get_stat_features(affected_features)
        # stat_unaffect = self.get_stat_features(unaffected_features)

        stat2 = torch.concat([stat_affect, stat_unaffect], dim=-2)  # [B, C, 18, 4]
        stat2 = self.fuse2(stat2)  # [b, embed, 18, 1]
        stat2 = torch.flatten(stat2, start_dim=1)  # [b, 18 * embed]

        logits = self.mlp_classify(stat2)

        if get_cl_loss:
            cl_loss = self.CL_Module(stat1, label, logits)
            return logits, cl_loss
        else:
            return logits


class STAT_Net_V3(nn.Module):
    def __init__(self, affect_idx: list, unaffected_idx: list, in_channels, embed_dim, classes):
        super(STAT_Net_V3, self).__init__()

        self.joints = len(affect_idx)
        self.affect = affect_idx
        self.unaffect = unaffected_idx

        self.num_class = classes

        self.VA_Block = STAT_VA(
            in_channels=in_channels,
            joints=self.joints
        )

        self.fuse1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, self.joints),
            stride=1,
            padding=0
        )

        self.fuse2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, self.joints),
            stride=1,
            padding=0
        )

        self.mlp_classify = nn.Sequential(
            # nn.Linear(in_features=16 * embed_dim, out_features=hidden),
            # nn.Dropout(0.3),
            # nn.ReLU(),
            # nn.Linear(in_features=hidden, out_features=self.num_class)

            nn.Linear(in_features=18 * embed_dim, out_features=self.num_class)
        )

        self.CL_Module = CL_Module(
            channels=embed_dim * 9,
            classes=self.num_class,
            use_p_map=True
        )

    def get_stat_features(self, x, proto=True):
        """
        :param x:     features in shape [b, c, f, j]
        :return:
                    calculate statistical information for each joint of each channel
                    in this way, we can squeeze temporal information
                    [b, c, 8, j]   '8' stands for 8 basic statistical values
        """
        s1, _ = torch.min(x, dim=-2, keepdim=True)
        s2, _ = torch.max(x, dim=-2, keepdim=True)
        s3 = torch.quantile(x, dim=-2, q=torch.tensor([0.25]).to(x.device), keepdim=True)[0]
        s4, _ = torch.median(x, dim=-2, keepdim=True)
        s5 = torch.quantile(x, dim=-2, q=torch.tensor([0.75]).to(x.device), keepdim=True)[0]
        s6 = torch.mean(x, dim=-2, keepdim=True)
        s7 = torch.std(x, dim=-2, keepdim=True)
        s8 = s2 - s1  # peak-to-peak
        s9_V1 = torch.mean(torch.pow((x - s6) / s7, 3), dim=-2, keepdim=True)       # Skewness
        s9_V2 = torch.mean(torch.pow((x - s6) / s7, 4), dim=-2, keepdim=True) - 3   # Kurtosis

        # diff_x = torch.zeros_like(x, device=x.device)
        diff_x = torch.clone(x.detach()).to(x.device)
        diff_x[:, :, 1:, :] = x[:, :, :-1, :]
        diff = x - diff_x
        s_10 = torch.norm(input=diff, p=1, dim=-2, keepdim=True)  # Total Variatuon

        if proto:
            stat_features = torch.concat([s2, s3, s4, s5, s6, s7, s8, s_10, s9_V1], dim=-2)
        else:
            stat_features = torch.concat([s2, s3, s4, s5, s6, s7, s8, s_10, s9_V2], dim=-2)

        return stat_features

    def forward(self, features, label=None, get_cl_loss=False, names=None, attn=True):
        unaffected_features = features[:, :, :, self.unaffect]
        affected_features = features[:, :, :, self.affect]

        stat_affect = self.get_stat_features(affected_features, proto=False)        # [B, C, 9, 4]
        stat_unaffect = self.get_stat_features(unaffected_features, proto=False)    # [B, C, 9, 4]
        mask = stat_unaffect - stat_affect

        if attn:
            unaffected_features = unaffected_features + self.VA_Block(mask, unaffected_features)
            affected_features = affected_features + self.VA_Block(mask, affected_features)

        stat1 = self.get_stat_features(unaffected_features + affected_features)
        stat1 = self.fuse1(stat1)                                                   # [B, embed, 9, 1]
        stat1 = torch.flatten(stat1, start_dim=1)                                   # [B, 9 * embed]

        """
        用attention之后的特征取stat values
        """
        # stat_affect = self.get_stat_features(affected_features)
        # stat_unaffect = self.get_stat_features(unaffected_features)

        stat2 = torch.concat([stat_affect, stat_unaffect], dim=-2)                  # [B, C, 18, 4]
        stat2 = self.fuse2(stat2)                                                   # [B, embed, 18, 1]
        stat2 = torch.flatten(stat2, start_dim=1)                                   # [B, 18 * embed]

        logits = self.mlp_classify(stat2)

        if get_cl_loss:
            cl_loss = self.CL_Module(stat1, label, logits)
            return logits, cl_loss
        else:
            return logits