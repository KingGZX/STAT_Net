import torch
import torch.nn as nn
from model.ST_STAT_GCN import ST_STAT_GCN

class STAT_Net(nn.Module):
    def __init__(self, num_class, item, attn=True) -> None:
        super().__init__()
        self.num_class = num_class
        self.item = item
        self.stat_attn = attn
        self.displacement_modal = ST_STAT_GCN(
            in_channels=9,
            num_class=self.num_class,
            groups=3,
            affected_idx=[14, 15, 16, 17],
            unaffected_idx=[18, 19, 20, 21],
            item=self.item,
            channels=[24, 48, 48],
            stat_embed=24,
            attn=self.stat_attn
        )
        
        self.jointangle_modal = ST_STAT_GCN(
            in_channels=3,
            num_class=self.num_class,
            groups=1,
            affected_idx=[14, 15, 16, 17],
            unaffected_idx=[18, 19, 20, 21],
            item=self.item,
            channels=[6, 12, 12],
            stat_embed=8,
            attn=self.stat_attn
        )
        
    def forward(self, x, label=None, get_cl_loss=False, names=None):
        dis_data = x[:, :9, :, :]
        ang_data = x[:, -3:, :, :]
        
        if get_cl_loss:
            dis_out, cl_loss1 = self.displacement_modal(dis_data, label, get_cl_loss, names)
            ang_out, cl_loss2 = self.jointangle_modal(ang_data, label, get_cl_loss, names)
            return dis_out, ang_out, cl_loss1 + cl_loss2
        else:
            dis_out = self.displacement_modal(dis_data, label, get_cl_loss, names)
            ang_out = self.jointangle_modal(ang_data, label, get_cl_loss, names)
            return dis_out, ang_out
        
        