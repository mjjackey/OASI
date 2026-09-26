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
