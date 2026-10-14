import argparse
import os
import sys
import torch
import random
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# import wandb
import datetime

class EarlyStopping:
    def __init__(self, args, patience=2, delta=0, type='acc', path='../output'):
        """
        Args:
            args: command line arguments
            patience: how many epochs to wait when no improvement
            delta: minimum improvement threshold to count as progress
            path: file path to save best model state dict
            type: metric for monitoring, support ['loss', 'acc', 'macro_f1']
            loss: lower is better; acc/macro_f1: higher is better
        """
        self.args = args
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.type = type
        self.path = path

    def __call__(self, val_loss, val_metric, model, optimiser, scheduler, epoch, global_step):
        """
        val_loss: validation loss value (only for logging)
        val_metric: metric value used for early‑stop judgment
        model: pytorch model instance to save
        """
        if self.type == "loss":
            # loss is better when smaller, convert to maximization objective
            score = -val_metric
        elif self.type in ("acc", "macro_f1"):
            # acc / macro_f1 is better when larger
            score = val_metric
        else:
            raise ValueError(f"unsupported monitor type {self.type}")

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(self.path,val_loss, model, optimiser, scheduler, epoch, global_step)
        elif score < self.best_score + self.delta:
            # no valid improvement observed
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}, best {self.type}:{self.best_score:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # metric improved, update best and save checkpoint
            self.best_score = score
            self.save_checkpoint(self.path,val_loss, model, optimiser, scheduler,epoch, global_step)
            self.counter = 0

    def save_checkpoint(self, path, val_loss, model,optimiser, scheduler, epoch,global_step):
        """Save model state dict when monitored metric reaches new best. """
        print(f"Metric improved ({self.best_score:.4f}). Saving model ...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step
        }, path)
        self.val_loss_min = val_loss

def set_args(seed:int=42):
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='oasi')
    parser.add_argument('--dataset', type=str, default='goemotions')
    parser.add_argument('--batch_size', type=int, default=64)  # original: 64
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--encoder_type', type=str, default='roberta-base')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--weight_decay', type=float, default=0.001) # original: 0.001
    parser.add_argument('--patience', type=int, default=4) # 2  # original: 4
    # parser.add_argument('--alpha', type=float, default=0.05) #0.05, 0.1, 0.2 original: 0.4
    parser.add_argument('--warmup_epochs', type=int, default=0)
    # parser.add_argument('--temperature', type=float, default=0.2) #0.15,0.2,0.3 original: 0.5
    # parser.add_argument('--base_temperature', type=float, default=0.07)
    parser.add_argument('--temperature', type=float, default=0.07) # 0.05~0.1
    parser.add_argument('--lambda_contrast', type=float, default=0.1) # 0.1~1.0
    parser.add_argument('--tau_ov', type=float, default=0.3)
    parser.add_argument('--alpha_ov', type=float, default=5.0)
    parser.add_argument('--device_id', type=str, default="0")
    parser.add_argument('--save_dir', type=str, default='../output')
    # parser.add_argument("--start_epoch", type=int, default=4, help="resume starting epoch")
    parser.add_argument("--k_n", type=float, default=8.0, help="sigmoid steepness for negative overlap weight")
    parser.add_argument("--t_n", type=float, default=0.3, help="sigmoid threshold for negative Jaccard overlap")
    parser.add_argument("--k_p", type=float, default=8.0, help="sigmoid steepness for positive annotator agreement weight")
    parser.add_argument("--t_p", type=float, default=0.7, help="sigmoid threshold for positive soft_p cos similarity")

    args = parser.parse_args()
    args.device = torch.device("cuda:" + args.device_id if torch.cuda.is_available() else "cpu")
    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def wandb_para_search_strag(high_overlap_f1,macro_f1,many_f1,few_f1,medium_f1):
     # 基线阈值：使用你原来默认参数(λ=0.1, τ_ov=0.5, α=5.0)得到的指标作为下限
    base_many = 0.6899
    base_few = 0.5473
    base_medium = 0.5340
    drop_tol = 0.02  # 允许最大下跌幅度，超过就惩罚
    drop_tol_soft = 0.04   # medium宽松阈值，允许更大下跌

    composite_score = 0.8 * high_overlap_f1 + 0.2 * macro_f1

    # Hard constraint penalty
    penalty = 0.0
    if many_f1 < base_many - drop_tol:
        penalty += 0.2
    if few_f1 < base_few - drop_tol:
        penalty += 0.2

    # Soft constraint for medium：只在暴跌的时候才罚，小幅下跌不惩罚
    # if medium_f1 < base_medium - drop_tol_soft:
    #     penalty += 0.15    

    composite_score = composite_score - penalty
    return composite_score

def wandb_para_search_strag2(high_overlap_f1,macro_f1,many_f1,medium_f1,few_f1):
    composite_score = 0.8 * high_overlap_f1 + 0.2 * macro_f1
    penalty = 0.0
    penalty_coeff = 1.5

    # Constraint 1: gap between many and few
    max_allowed_gap_few = 0.22
    gap_few = many_f1 - few_f1
    if gap_few > max_allowed_gap_few:
        # 超出部分乘以惩罚系数
        penalty += (gap_few - max_allowed_gap_few) * penalty_coeff 

    # Constraint 2: gap between many and medium (new added)
    # max_allowed_gap_medium = 0.18
    # gap_medium = many_f1 - medium_f1
    # if gap_medium > max_allowed_gap_medium:
    #     penalty += (gap_medium - max_allowed_gap_medium) * penalty_coeff

    # 可选：many本身不能过低（防止many直接崩）
    min_many_f1 = 0.55
    if many_f1 < min_many_f1:
        penalty += (min_many_f1 - many_f1) * 2.0
    
    # Constraint4: absolute lower bound to prevent many total collapse
    min_few_f1 = 0.42
    if few_f1 < min_few_f1:
        penalty += (min_few_f1 - few_f1) * 2.0

    composite_score = composite_score - penalty
    return composite_score
