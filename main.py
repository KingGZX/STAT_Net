from config import Config
from loadata import Dataset
from baselines.ST_GCN_baseline import ST_GCN
from baselines.CTR_GCN import Model
from model.Model import STAT_Net
from utils.train import *
from train_stat import batch_train_stat
import pickle
import os
import torch



if __name__ == "__main__":
    
    cfg = Config()
    with open('./binary/dst1706.pkl', 'rb') as inp:
        dataset = pickle.load(inp)
    net = Model(
        in_channels=cfg.in_channels,
        num_class=3,
    )
    batch_train(
        model=net,
        dataset=dataset,
        epochs=30,
        model_name="CTR-GCN",
        cfg=cfg,
        items=[8],
        batchsize=16
    )