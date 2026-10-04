import argparse
import sys
import torch
import random
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# import wandb
import datetime
from description import ED_description, GE_description

DEBUG_K_M = False
class CrossEncoder(nn.Module):
    def __init__(self, args, num_labels, tkr, head='mlp', features_dim=128):
        super(CrossEncoder, self).__init__()
        self.args = args
        self.num_labels = num_labels
        self.encoder_type = args.encoder_type
        self.device = args.device
        self.alpha = args.alpha
        self.seed = args.seed
        self.temperature = args.temperature
        self.lambda_contrast = args.lambda_contrast
        self.warmup_epochs = args.warmup_epochs
        # self.base_temperature = args.base_temperature
        self.tau_ov= args.tau_ov
        self.alpha_ov = args.alpha_ov

        label_description = ED_description if args.dataset == 'ED' else GE_description
        max_seq_len = 128 if args.dataset == 'goemotions' else 64
        label_input_ids, label_attn_masks = [], []
        for label, description in label_description.items():
            label_tokens = tkr.tokenize(label)
            description_tokens = tkr.tokenize(description)

            tokens = label_tokens + ["<#>"] + description_tokens
            tokens = [tkr.cls_token] + tokens + [tkr.sep_token]
            attn_mask = [1] * len(tokens)
            if len(tokens) < max_seq_len:
                attn_mask += [0] * (max_seq_len - len(tokens))
                tokens += [tkr.pad_token] * (max_seq_len - len(tokens))

            input_ids = tkr.convert_tokens_to_ids(tokens)
            label_input_ids.append(input_ids)
            label_attn_masks.append(attn_mask)
        self.label_input_ids = torch.tensor(label_input_ids).to(args.device)
        self.label_attn_masks = torch.tensor(label_attn_masks).to(args.device)

        self.encoder = AutoModel.from_pretrained(         
                args.encoder_type, 
                hidden_dropout_prob=0.2,          # dropout inside transformer layers
                attention_probs_dropout_prob=0.2  # dropout for attention weight
        )
        print("hidden_dropout_prob:", self.encoder.config.hidden_dropout_prob)
        print("attention_probs_dropout_prob:", self.encoder.config.attention_probs_dropout_prob)
        self.encoder.resize_token_embeddings(len(tkr))

        self.config = self.encoder.config
        ## Equals look up table for label embeddings, rows = num_labels, cols = hidden_size
        self.label_embedding = nn.Embedding(num_labels, self.config.hidden_size) # rows, cols
        ## Mark parameters as trainable, so that they can be optimized during training 
        self.label_embedding_mat = nn.Parameter(torch.randn(num_labels, self.config.hidden_size,device=self.device), requires_grad=True)

        if head == 'linear':
            self.head = nn.Linear(self.config.hidden_size, features_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, features_dim)
            )
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

        # self.wloss = WLoss([self.alpha, self.gamma])
        # self.is_W_loss = False
        # self.constrative_criterion = SupClusterConLoss(self.device,
        #                                                self.temperature,
        #                                                self.base_temperature,
        #                                                0,
        #                                                len(label_input_ids),
        #                                                self.alpha,
        #                                                self.gamma)

    def forward(self, input_ids=None, attention_mask=None, labels=None, soft_p_tensor=None,lbl_set_list=None,current_epoch=0):
        """
        Args:
            input_ids: token_id, shape: [batch_size, seq_len] 
            attention_mask: 1 or 0, shape: [batch_size, seq_len] 
            labels: label index, shape: [batch_size]
            soft_p_tensor: soft probability tensor, shape: [batch_size, num_classes]
        Parameters get from model(input_ids, attention_mask, labels) in train() which gets form dataloader
        """
        # Get projection head, last_hidden_state, shape: [batch_size, seq_len, hidden_size]  
        # --> CLS token representation[batch_size, hidden_size]-->[batch_size,out_dim]
        features = self.head(self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :])
        # output shape: [num_labels, hidden_size]
        label_emb = self.encoder(input_ids=self.label_input_ids, attention_mask=self.label_attn_masks)[0][:, 0, :]
        # output shape: [num_labels, out_dim]
        features_label = self.head(label_emb)

        # shape: [batch_size, num_lables]
        lbl_logits = torch.matmul(features, (features_label).T)
        cls_loss = F.cross_entropy(lbl_logits, labels, reduction='mean')

        features_norm = F.normalize(features, dim=1)
        # mask, logits_mask, loss_valid_mask = self.sup_kmeans(features_norm.detach().cpu().numpy(),
        #                                     labels.detach().cpu().numpy())

        # if self.is_W_loss:
        #     loss_combined, loss_ldl, loss_ccl = self.constrative_criterion(features, labels,
        #                                        label_emb, self.num_labels,
        #                                        mask, logits_mask, lbl_logits, loss_valid_mask)
        #     contra_loss = self.wloss([loss_ccl, loss_ldl])
        # else:
        #     sys.exit()
        if current_epoch < self.warmup_epochs:
            contra_loss = torch.tensor(0.0, device= self.args.device)
        else:
            contra_loss = self.soft_info_nce_loss(features_norm, soft_p_tensor, labels, lbl_set_list, self.temperature,self.tau_ov,self.alpha_ov)

        # if __debug__:
        #     print(f"cls_loss: {cls_loss.item()}, contrastive loss: {contra_loss.item()}")
        total_loss = cls_loss + self.lambda_contrast * contra_loss

        return lbl_logits, cls_loss, contra_loss, total_loss


    def soft_info_nce_loss(self,
        features_norm: torch.Tensor,
        soft_p: torch.Tensor,
        labels: torch.Tensor,
        lbl_set_list: list,
        tau: float = 0.07,
        tau_ov: float = 0.3,
        alpha_ov: float = 5.0,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Soft‑InfoNCE with overlap‑aware filtering (single‑label setting)
        overlap‑aware: only same‑label samples are positive candidates; different‑label are negatives.
        Positive pair weight w_ij = cos(soft_p[i], soft_p[j]) from annotator distribution.

        Args:
            features_norm: torch.Tensor, shape [B, D], L2‑normalized projection head
            soft_p: torch.Tensor, shape [B, num_classes], annotator distribution
            lbl_set_list: list of label set S_i for each sample in batch
            labels: torch.Tensor, shape [B], ground‑truth label index (long)
            tau_o: jaccard overlap threshold
            alpha: decay factor for overlap penalty
            eps: avoid division by zero

        Returns:
            loss: scalar torch.Tensor
        """
        assert isinstance(labels, torch.Tensor), f"labels should be tensor, got {type(labels)}"
        assert labels.dim() == 1, f"labels expected shape [B], got {labels.shape}"
        B = features_norm.size(0)
        assert labels.size(0) == B, "batch size mismatch"

        # 1. compute similarity between text embeddings: z_i · z_j  (since normalized)
        sim_z = torch.matmul(features_norm, features_norm.T)  # [B, B]

        # 2. weight matrix w_ij: cosine similarity of annotator distribution p_i, p_j
        w_annotator = F.cosine_similarity(
            soft_p.unsqueeze(1),   # [B,1,27]
            soft_p.unsqueeze(0),   # [1,B,27]
            dim=-1
        )  # [B,B]

        # Precompute Jaccard overlap o_ij [B,B]
        jac_ov_sim = torch.zeros((B,B), dtype=torch.float32, device=features_norm.device)
        for i in range(B):
            for j in range(B):
                si = lbl_set_list[i]
                sj = lbl_set_list[j]
                inter = len(si & sj)
                union = len(si | sj)
                jac_ov_sim[i,j] = inter / union if union > 0 else 0.0
           
        # Overlap-aware soft weight w_ov
        w_ov = torch.where(
            jac_ov_sim <= tau_ov,
            torch.ones_like(jac_ov_sim),
            torch.exp(-alpha_ov * jac_ov_sim)
        )

        # Total weight: annotator similarity * overlap penalty
        # w_total = w_annotator * w_ov

        # 3. overlap‑aware mask: positive candidate mask (same label)
        pos_candidate_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # [B,B] 
        # exclude self‑pair i == i
        self_mask = 1.0 - torch.eye(B, device = features_norm.device)
        # for i in range(B):
        #     for j in range(B):
        #         # The raw label set of sample i is totaly equal to the label set of sample j
        #         if i != j and lbl_set_list[i].issubset(lbl_set_list[j]) and lbl_set_list[j].issubset(lbl_set_list[i]):
        #             pos_candidate_mask[i,j] = True
        pos_candidate_mask = pos_candidate_mask * self_mask

        exp_sim = torch.exp(sim_z / tau)

        # numerator: sum over valid positive pairs: w_total * exp(sim_ij/tau)
        numerator = torch.sum(exp_sim * w_annotator * pos_candidate_mask.float(), dim=-1)  # [B]

        # denominator: sum over all j≠i (all candidates, positive + negative)
        denominator = torch.sum(exp_sim * w_ov * self_mask.float(), dim=-1)  # [B]

        # InfoNCE per‑sample loss
        per_sample_loss = -torch.log((numerator + eps) / (denominator + eps))
        loss = torch.mean(per_sample_loss)
        # sum_loss = torch.sum(per_sample_loss)
        # print(f"loss: {loss.item()}, sum_loss: {sum_loss.item()}")  # debug

        return loss
