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
from utils import SupClusterConLoss
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
        self.gamma = args.gamma
        self.seed = args.seed
        self.temperature = args.temperature
        self.lambda_contrast = args.lambda_contrast
        self.warmup_epochs = args.warmup_epochs
        # self.base_temperature = args.base_temperature

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

    def forward(self, input_ids, attention_mask, labels, soft_p_tensor,lbl_set_list,current_epoch=0):
        """
        Args:
            input_ids: token_id, shape: [batch_size, seq_len] 
            attention_mask: 1 or 0, shape: [batch_size, seq_len] 
            labels: label index, shape: [batch_size]
            soft_p_tensor: soft probability tensor, shape: [batch_size, num_classes]
        Parameters get from model(input_ids, attention_mask, labels) in train() which gets form dataloader
        """
        # Get projection head, last_hidden_state, shape: [batch_size, seq_len, hidden_size] --> CLS token representation[batch_size, hidden_size]-->[batch_size,out_dim]
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
            contra_loss = self.soft_info_nce_loss(features_norm, soft_p_tensor, labels, lbl_set_list, self.temperature)

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
        o_mat = torch.zeros((B,B), dtype=torch.float32, device=z.device)
        for i in range(B):
            for j in range(B):
                si = lbl_set_list[i]
                sj = lbl_set_list[j]
                inter = len(si & sj)
                union = len(si | sj)
                o_mat[i,j] = inter / union if union > 0 else 0.0

        # 3. overlap‑aware mask: positive candidate mask (same label)
        pos_candidate_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # [B,B]
        # exclude self‑pair i == i
        self_mask = 1.0 - torch.eye(B, device = features_norm.device)
        pos_candidate_mask = pos_candidate_mask * self_mask

        exp_sim = torch.exp(sim_z / tau)

        # numerator: sum over valid positive pairs: w_ij * exp(sim_ij/tau)
        numerator = torch.sum(exp_sim * w_annotator * pos_candidate_mask, dim=-1)  # [B]

        # denominator: sum over all j≠i (all candidates, positive + negative)
        denominator = torch.sum(exp_sim * self_mask, dim=-1)  # [B]

        # InfoNCE per‑sample loss
        per_sample_loss = -torch.log((numerator + eps) / (denominator + eps))
        loss = torch.mean(per_sample_loss)

        return loss


    def get_features(self, input_ids, attention_mask):
        features = self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0, :]
        # feats_norm = F.normalize(self.head(feats), dim=1)
        features_norm = self.head(features)

        return features_norm

    def sup_kmeans(self, features, labels):
        """
        Args: 
            features: [batch_size, out_dim]
            labels: label index, shape: [batch_size] 
        """
        cls_num, batch_size = len(np.unique(labels)), features.shape[0]
        
        ############################ Search the optimal cluster number opt_M  ##########################
        if DEBUG_K_M:
            min_J, opt_M = 1e6, 0

            for M in tqdm(range(cls_num, cls_num + 10)):
                kmeans = KMeans(n_clusters=M, init='k-means++', random_state=0)
                kmeans.fit_predict(features)
            
                purity, purity_err_rate = 0,0
                # shape: [opt_M, label_num]
                P = np.zeros(opt_M, len(labels))
                for cluster_id in range(M):
                    ## Get the indices of all samples in the current cluster_id cluster
                    indices = np.where(kmeans.labels_ == cluster_id)[0]
                    ## Extract the true labels of all samples within the current cluster and their counts
                    unique_labels, counts = np.unique(labels[indices], return_counts=True)
                    ## The total number of "primary class" samples within all clusters 
                    ## (clustering purity metric; the larger the value, the more closely the clustering results match the true labels)
                    purity += np.max(counts)   
                    ## Get P[i,j] = P({true label}=j | sample belongs to cluster i))
                    P = P / np.sum(P, axis=1, keepdims=True)
                    # shape: [opt_M]
                    purity_per_cluster = np.max(P, axis=1)

                ## Convert to purity error rate
                purity_err_rate = (batch_size - purity) / batch_size
                J = purity_err_rate + np.sqrt((M - cls_num) / batch_size)
                if J < min_J:
                    min_J = J
                    opt_M = M
        ###############################################################################################

        opt_M = cls_num
        kmeans = KMeans(n_clusters=opt_M, init='k-means++', random_state=self.seed)
        kmeans.fit_predict(features)
        cluster_labels = kmeans.labels_
        
        ## Initialize mask(positive sign) and logits_mask(take part in contrastive loss caclutation sign) 
        mask, logits_mask = torch.zeros(batch_size, batch_size, device=self.device), torch.zeros(batch_size, batch_size, device=self.device)
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                ## Hard negative: Intra-class instances, purpose: split, increase intra-class variation 
                if cluster_labels[i] == cluster_labels[j] and labels[i] != labels[j]:
                    logits_mask[i, j] = logits_mask[j, i] = 1
                ## Hard positive: Inter-class instances, purpose: aggregation 
                elif cluster_labels[i] != cluster_labels[j] and labels[i] == labels[j]:
                    logits_mask[i, j] = logits_mask[j, i] = 1
                    mask[i, j] = mask[j, i] = 1

        ### Prevent the condition where is no negative samples result in loss=0
        labels = torch.tensor(labels, device=self.device)
        B = logits_mask.shape[0]
        ## loss_valid_mask is for conditions where there is no negative sample for a given sample, which would result in loss=0.
        loss_valid_mask = torch.ones(B, dtype=torch.bool, device=logits_mask.device)
        # Get the sample's indice array which the whole row logits_mask is zero
        zero_mask_idx = torch.nonzero(logits_mask.sum(1) == 0).squeeze(1)
        for idx in zero_mask_idx:
            # Gather all negative sample indices (for different labels)
            neg_sample_idices = torch.nonzero(~(labels == labels[idx])).squeeze(1)
            if  neg_sample_idices.shape[0] == 0:
            # No negative samples, set False, this sample will not take part in loss calculation  
                loss_valid_mask[idx] = False
                continue  
            indices = torch.randperm(neg_sample_idices.shape[0])
            num_neg = min(3, neg_sample_idices.shape[0])
            selected_neg = neg_sample_idices[indices[:num_neg]]
            logits_mask[idx, selected_neg] = 1

        return mask, logits_mask, loss_valid_mask


class WLoss(nn.Module):
    def __init__(self, weights):
        super(WLoss, self).__init__()
        self.weights = weights

    def forward(self, inputs):
        total_loss = 0
        for i in range(len(self.weights)):
            total_loss += self.weights[i] * inputs[i]
        return total_loss