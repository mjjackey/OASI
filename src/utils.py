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
    def __init__(self, args, patience=2, delta=0, path='checkpoint.pt', type='acc'):
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
        self.path = path
        self.type = type

    def __call__(self, val_loss, val_metric, model):
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
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # no valid improvement observed
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}, best {self.type}:{self.best_score:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # metric improved, update best and save checkpoint
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model state dict when monitored metric reaches new best. """
        print(f"Metric improved ({self.best_score:.4f}). Saving model ...")
        # torch.save(model.state_dict(), self.path)
        ckpt_name = f"best_seed{self.args.seed}_lc{self.args.lambda_contrast}_tau{self.args.temperature}.pt"
        torch.save(model.state_dict(), os.path.join(self.args.save_dir, ckpt_name))
        self.val_loss_min = val_loss


class SupClusterConLoss(nn.Module):
    def __init__(self, device, temperature, base_temperature, beta, label_num, alpha, gamma):
        super(SupClusterConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device
        self.beta = beta
        self.label_num = label_num
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, features, labels, lbl_emb, num_labels, positive_mask, logits_mask, lbl_logits, loss_valid_mask):
        """
        Args:
            features:  batch sample features [B, out_dim]
            labels:    batch ground‑truth labels [B]
            lbl_emb:   learnable label embedding matrix [num_labels, out_dim]
            num_labels: total number of classes
            positive_mask:  positive pair mask [B, B], mask[i,j]=1 means j is positive for i
            logits_mask: contrastive denominator mask [B, B]
            loss_valid_mask: [B] bool, True: sample can compute loss; False: skip sample (denominator empty)
        Returns:
            total_loss: combined loss
            exp_ldl_loss: exp of ldl loss 
            supcon_loss_mean: pure supervised‑contrastive loss
        """
        ## Force the loss calculation to be performed in float32 to avoid overflow of exp/log in bf16 mode
        features = features.float()
        lbl_emb = lbl_emb.float()
        lbl_logits = lbl_logits.float()
        positive_mask = positive_mask.float()
        logits_mask = logits_mask.float()
        loss_valid_mask = loss_valid_mask.bool()

        feats_norm = F.normalize(features, dim=1)
        lbl_emb_norm = F.normalize(lbl_emb, dim=1)

        # ========= label‑aware disentangled loss (LDL), pairwise confusability score  =========
        # [B, num_labels] is M matrix of the paper,  [num_labels, B], each row is M's column M[:,j]
        lbl_sim_emb = lbl_logits.T
        # compute pairwise confusability score f(e_j, e_j')
        confusion_cos= F.cosine_similarity(lbl_sim_emb.unsqueeze(1).expand(-1, num_labels, -1),
                                      lbl_sim_emb.unsqueeze(0).expand(num_labels, -1, -1), dim=-1)
        triu_mask = torch.triu(torch.ones_like(confusion_cos), diagonal=1).bool()
        # upper‑triangular confusability score (f(e_j,e_j'))
        upper_confu = confusion_cos[triu_mask]
        # normalize to sum‑to‑one → weights f
        weights = upper_confu / (upper_confu.sum() + 1e-8)

        # pairwise cosine between label‑embedding lbl_emb_norm [num_labels,H]
        lbl_cos_mat =  F.cosine_similarity(lbl_emb_norm.unsqueeze(1).expand(-1, num_labels, -1),
                                      lbl_emb_norm.unsqueeze(0).expand(num_labels, -1, -1), dim=-1)
        upper_lbl_cos = lbl_cos_mat[triu_mask]
    
        # LDL loss: sum( f(e_j,e_j') * cos(l_j, l_j') ), minimize
        loss_ldl = torch.sum(weights * upper_lbl_cos)
        loss_ldl_final = self.gamma * loss_ldl  
        exp_ldl_loss = torch.exp(loss_ldl)

        # ========== Supervised‑Contrastive Loss (SupCon) ==========
        # [B, B]
        anchor_dot_contrast = torch.div(
            torch.matmul(feats_norm, feats_norm.T),
            self.temperature)
        # logits_max's shape: [B,1]
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        feature_logits = anchor_dot_contrast - logits_max.detach()
        # mask = mask.repeat(feats_norm, feats_norm)  # (1, 1)
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * feats_norm).view(-1, 1).to(device),
        #     0     
        # )
        exp_logits = torch.exp(feature_logits) * logits_mask  #[B, B]
        log_prob = feature_logits - torch.log(exp_logits.sum(1, keepdim=True)) #[B, B]
        pos_sum = (positive_mask * log_prob).sum(1) # [B]
        mask_row_sum = positive_mask.sum(1)         # [B]
        ## mark which rows contain positive samples, avoid division by zero
        valid_row = mask_row_sum > 0
        mean_log_prob_pos = torch.zeros_like(pos_sum)
        ## work for valid rows
        mean_log_prob_pos[valid_row] = pos_sum[valid_row] / mask_row_sum[valid_row]
        ## invalid rows(no positive samples) set to zero, not taken into account for loss calculation
        mean_log_prob_pos[~valid_row] = 0.0

        # apply loss_valid_mask: skip samples with empty denominator(logits_mask all zero)
        mean_log_prob_pos = mean_log_prob_pos[loss_valid_mask]

        if mean_log_prob_pos.numel() == 0:
            # all samples are invalid, avoid empty tensor mean() → nan
            ccl_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        else:
            ccl_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
            ccl_loss = ccl_loss.mean()

        total_loss = loss_ldl_final + self.alpha * ccl_loss
        if __debug__:
            print(f"clustering-guided contrastive loss: {ccl_loss}")

        return total_loss, exp_ldl_loss, ccl_loss