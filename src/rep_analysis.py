from collections import Counter
import csv
import os
import random
import numpy as np
import torch
from torch.cuda.random import seed_all
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from datasets import get_dicts, emotion_DataSet
from models import CrossEncoder
from utils import set_args,set_seed
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import umap
# import wandb


target_confuse_labels = {"admiration", "approval", "sadness", "disappointment", "fear", "nervousness"}
    
def load_model(ckpt_path,args,num_classes,tkr):
    """Load model checkpoint and set to evaluation mode."""
    model = CrossEncoder(args,num_classes,tkr)
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    # If your ckpt only saves state dict, replace with: model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    # model.eval()

    return model

def get_preds_labels(model,val_dataloader,device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader: # val / test dataloader
            # TODO
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            logits,_,_,_ = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_labels,all_preds

def calc_per_class_f1(all_labels, all_preds,num_classes,train_class_counts,filepath='per_class_f1'):
    # ===================== 【2】计算逐类别F1 =====================
    
    per_class_f1 = f1_score(all_labels, all_preds, labels=list(range(num_classes)), average=None)

    # Print per-class F1
    # print("\n===== Per-class F1 Score =====")
    # for cls_id in range(num_classes):
    #     print(f"Class {cls_id:02d} | F1 = {per_class_f1[cls_id]:.4f}")

    # Full classification report
    # print("\nClassification Report:")
    # print(classification_report(all_labels, all_preds, digits=4))

    # ===================== 【3】长尾分组：Many / Medium / Few  =====================
    # !!! Important: train_class_counts should be precomputed from TRAIN set
    # Example: train_class_counts = [number of samples for class 0, class1, ..., class26]
    train_class_counts = np.array(train_class_counts)

    # Split by training set sample count (common rule: 3 quantiles)
    quantiles = np.quantile(train_class_counts, [0.33, 0.66])
    many_cls_idx = np.where(train_class_counts > quantiles[1])[0]
    medium_cls_idx = np.where((train_class_counts >= quantiles[0]) & (train_class_counts <= quantiles[1]))[0]
    few_cls_idx = np.where(train_class_counts < quantiles[0])[0]

    many_f1_mean = np.mean(per_class_f1[many_cls_idx])
    medium_f1_mean = np.mean(per_class_f1[medium_cls_idx])
    few_f1_mean = np.mean(per_class_f1[few_cls_idx])

    print(f"\n===== Long-tailed Group F1 =====")
    print(f"Many-shot classes ({len(many_cls_idx)}): Avg F1 = {many_f1_mean:.4f}")
    print(f"Medium-shot classes ({len(medium_cls_idx)}): Avg F1 = {medium_f1_mean:.4f}")
    print(f"Few-shot classes ({len(few_cls_idx)}): Avg F1 = {few_f1_mean:.4f}")

    if 'Test' in filepath:
        group_avg_f1 = np.array([ many_f1_mean,  medium_f1_mean, few_f1_mean])
        # concatenate to one row：27 + 3 = 30 elements
        row_all = np.concatenate([per_class_f1, group_avg_f1])
        np.save(f"{filepath}_f1_one_row.npy", row_all)

    # ===================== 【4】W&B log（你在用wandb，直接上报） =====================
    # if wandb != None:
    #     wandb.log({
    #         "eval/many_f1": many_f1_mean,
    #         "eval/medium_f1": medium_f1_mean,
    #         "eval/few_f1": few_f1_mean,
    #     })
    
    return per_class_f1, many_f1_mean,medium_f1_mean,few_f1_mean


def get_embeddings(model, test_dataset, dataloader, device, id2name):
    """Extract projected contrastive embeddings (inference only, no gradient)."""
    model.eval()
    embs = []
    gold_labels = []
    is_confusing_mask = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            feats = model(input_ids, attention_mask, return_features=True)
            embs.append(feats.cpu())
            # TODO
            gold_labels.append(batch[2].cpu().numpy())

    # Build confusing mask by iterating test dataset (raw_label_ids)
    for idx in range(len(test_dataset)):
        raw_label_ids = test_dataset[idx]["raw_label_ids"]
        raw_names = set([id2name[lid] for lid in raw_label_ids])
        is_confusing_mask.append(True if (raw_names & target_confuse_labels) else False)

    return torch.vstack(embs), np.hstack(gold_labels), np.array(is_confusing_mask)


def umap_project(embeddings, n_neighbors=15, min_dist=0.1, random_state=42):
    """Project high-dimensional embeddings to 2D space with UMAP."""
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=random_state, metric="cosine")
    proj_2d = reducer.fit_transform(embeddings.numpy())
    return proj_2d


SUPCONBASE = True
FULLMODE = False
def get_model_dataset(prefix:str,seed:int):
    args = set_args(seed)
    set_seed(args)

    _, id2name = get_dicts('goemotions')
    num_classes = len(id2name.items())
    
    tkr = AutoTokenizer.from_pretrained("roberta-base")
    special_tokens_dict = {'additional_special_tokens': ["<#>"]}
    tkr.add_special_tokens(special_tokens_dict)

    # if FULLMODE:
    #     prefix = "full"
    # elif SUPCONBASE:
    #     prefix = "sup"
    if "sig" in prefix: 
        ckpt_name = f"{prefix}_best_seed{args.seed}_lc{args.lambda_contrast}_tau{args.temperature}_t_n{args.t_n}_t_p{args.t_p}.pt" 
    else:
        ckpt_name = f"{prefix}_best_seed{args.seed}_lc{args.lambda_contrast}_tau{args.temperature}_tau_ov{args.tau_ov}.pt"
    ckpt_path = os.path.join(args.save_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"WARNING: ckpt not found, skip {ckpt_path}")
        return None,None,None
    model = load_model(ckpt_path,args,num_classes,tkr)
    
    testset = emotion_DataSet('test', tkr, args.dataset)
    # CRITICAL: keep False for index alignment
    # TODO
    test_loader = DataLoader(testset, batch_size=8, shuffle=False,
                              num_workers=0, pin_memory=True, collate_fn=testset.collate)
    
    return model,test_loader,args.device


def get_train_dataset():
    args = set_args()
    set_seed(args)

    _, id2name = get_dicts('goemotions')
    num_classes = len(id2name.items())

    tkr = AutoTokenizer.from_pretrained("roberta-base")
    special_tokens_dict = {'additional_special_tokens': ["<#>"]}
    tkr.add_special_tokens(special_tokens_dict)

    train_dataset = emotion_DataSet('train',tkr,args.dataset)
    train_labels = [sample[1] for sample in train_dataset]
    train_class_counts = Counter(train_labels)
    # sort to 0~26 order
    train_class_counts = [train_class_counts[i] for i in range(num_classes)]

    # Split by training set sample count (common rule: 3 quantiles)
    quantiles = np.quantile(train_class_counts, [0.33, 0.66])
    many_cls_idx = np.where(train_class_counts > quantiles[1])[0]
    medium_cls_idx = np.where((train_class_counts >= quantiles[0]) & (train_class_counts <= quantiles[1]))[0]
    few_cls_idx = np.where(train_class_counts < quantiles[0])[0]
    
    return train_class_counts,train_dataset,num_classes,many_cls_idx,medium_cls_idx,few_cls_idx


def plot_umap():
    model,test_loader,device = get_model_dataset()
    emb, gold_lbls, mask = get_embeddings(model,test_loader,device)
    proj = umap_project(emb)
    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4.5), dpi=300)
    # SupInfoNCE subplot
    ax1.scatter(proj[~mask,0], proj[~mask,1], c="#cccccc", s=7, alpha=0.4)
    sc1 = ax1.scatter(proj[mask,0], proj[mask,1], c=gold_lbls[mask], s=7, alpha=0.8, cmap="tab20")
    ax1.set_title("SupInfoNCE", fontsize=12)
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    plt.savefig("umap.pdf", bbox_inches="tight")
    plt.close()


def get_per_class_f1(prefix:str,seed):
    model,test_loader,device = get_model_dataset(prefix,seed)
    train_class_counts, _ ,num_classes,_,_,_ = get_train_dataset()
    all_labels, all_preds = get_preds_labels(model,test_loader,device)
    return calc_per_class_f1(all_labels,all_preds,num_classes,train_class_counts)


def compute_classwise_avg_jaccard(train_dataset, num_classes=27):
    """
    输入：训练集dataset，每个样本包含 label（单标签） + lbl_set（原始多标签集合）
    返回：class_avg_jaccard: array[27], class_avg_jaccard[c] = 类别c内部样本两两Jaccard均值
    """
    # 1. 按类别收集所有样本的原始label_set
    cls_label_sets = [[] for _ in range(num_classes)]
    for sample in train_dataset:
        # TODO
        c = sample[1]  # 单标签
        lbl_set = sample[3] # Python set，原始多标签集合
        cls_label_sets[c].append(lbl_set)

    class_avg_jaccard = np.zeros(num_classes)
    for c in range(num_classes):
        set_list = cls_label_sets[c]
        n = len(set_list)
        if n <= 1:
            # 类别只有1个样本，无样本对，置0
            class_avg_jaccard[c] = 0.0
            continue
        total_jac = 0.0
        pair_cnt = 0
        # 遍历类别内全部两两样本对
        for i in range(n):
            for j in range(i+1, n):
                s_i = set_list[i]
                s_j = set_list[j]
                inter = len(s_i & s_j)
                union = len(s_i | s_j)
                jac = inter / union if union > 0 else 0.0
                total_jac += jac
                pair_cnt +=1
        class_avg_jaccard[c] = total_jac / pair_cnt
    return class_avg_jaccard


def get_classwise_avg_jaccard():
    _, train_dataset, num_classes,_,_,_ = get_train_dataset()
    class_avg_jaccard = compute_classwise_avg_jaccard(train_dataset,num_classes=num_classes)
    filename = "class_avg_jaccard.npy"
    np.save(f"../output/{filename}", class_avg_jaccard)
    return class_avg_jaccard


def get_1seed_overlap_f1(per_class_f1,class_avg_jaccard):
    # per-class F1 in order class0 ~ class26
    # per_class_f1 = np.array([
    # 0.7790, 0.8802, 0.5387, 0.4057, 0.5932, 0.5868, 0.5165,
    # 0.7292, 0.6471, 0.4076, 0.5656, 0.5860, 0.6512, 0.4839,
    # 0.7704, 0.9159, 0.0000, 0.6462, 0.8284, 0.6923, 0.6432,
    # 0.3636, 0.3308, 0.6667, 0.6327, 0.6571, 0.5829
    # ])

    # ========== load class_avg_jaccard ==========
    # class_avg_jaccard = np.load("../output/class_avg_jaccard.npy")
    jac_med = np.median(class_avg_jaccard)
    high_mask = class_avg_jaccard >= jac_med
    low_mask = class_avg_jaccard < jac_med

    high_overlap_f1_mean = np.mean(per_class_f1[high_mask])
    low_overlap_f1_mean = np.mean(per_class_f1[low_mask])

    print(f"High-overlap classes Avg F1: {high_overlap_f1_mean:.4f}")
    print(f"Low-overlap classes Avg F1: {low_overlap_f1_mean:.4f}")

    return high_overlap_f1_mean,low_overlap_f1_mean


def get_1seed_cross_tailed_jac_f1():
    _,_,_,many_cls_idx,medium_cls_idx,few_cls_idx=get_train_dataset()

    class_avg_jaccard = np.load("../output/class_avg_jaccard.npy")
    jac_med = np.median(class_avg_jaccard)
    high_mask = class_avg_jaccard >= jac_med
    low_mask = class_avg_jaccard < jac_med

    per_class_f1 = np.array([
        0.7790, 0.8802, 0.5387, 0.4057, 0.5932, 0.5868, 0.5165,
        0.7292, 0.6471, 0.4076, 0.5656, 0.5860, 0.6512, 0.4839,
        0.7704, 0.9159, 0.0000, 0.6462, 0.8284, 0.6923, 0.6432,
        0.3636, 0.3308, 0.6667, 0.6327, 0.6571, 0.5829
        ])

    # ========== 2. Build 2 × 3 cross grouping mask ==========
    # create boolean mask for each long-tail group
    is_many = np.zeros(27, dtype=bool)
    is_many[many_cls_idx] = True

    is_medium = np.zeros(27, dtype=bool)
    is_medium[medium_cls_idx] = True

    is_few = np.zeros(27, dtype=bool)
    is_few[few_cls_idx] = True

    # cross mask: overlap condition AND long-tail condition
    # row: high / low overlap; column: many / medium / few
    cell_masks = [
        # High overlap
        [high_mask & is_many,
        high_mask & is_medium,
        high_mask & is_few],
        # Low overlap
        [low_mask & is_many,
        low_mask & is_medium,
        low_mask & is_few]
    ]

    # ========== 3. Calculate average per-class F1 for each cell ==========
    result_table = np.zeros((2,3))
    cell_count = np.zeros((2,3), dtype=int)

    for i in range(2):
        for j in range(3):
            mask = cell_masks[i][j]
            selected_f1 = per_class_f1[mask]
            if len(selected_f1) > 0:
                result_table[i,j] = np.mean(selected_f1)
                cell_count[i,j] = len(selected_f1)
            else:
                result_table[i,j] = np.nan
                cell_count[i,j] = 0

    # ========== 4. Print results ==========
    row_names = ["High-overlap", "Low-overlap"]
    col_names = ["Many-shot", "Medium-shot", "Few-shot"]
    print("=== Cross Group: Overlap × Long-tail, Avg Per-class F1 ===")
    print(f"{'':<15}{col_names[0]:<12}{col_names[1]:<12}{col_names[2]:<12}")
    for i in range(2):
        line = f"{row_names[i]:<15}"
        for j in range(3):
            line += f"{result_table[i,j]:.4f} ({cell_count[i,j]}cls)    "
        print(line)

    print("\nCell count matrix (number of categories):")
    print(cell_count)

def save_oasi_f1():
    # seed=42
    seed42_f1 = np.array([
        0.7790, 0.8802, 0.5387, 0.4057, 0.5932, 0.5868, 0.5165,
        0.7292, 0.6471, 0.4076, 0.5656, 0.5860, 0.6512, 0.4839,
        0.7704, 0.9159, 0.0000, 0.6462, 0.8284, 0.6923, 0.6432,
        0.3636, 0.3308, 0.6667, 0.6327, 0.6571, 0.5829
    ])

    # seed=123
    seed123_f1 = np.array([
        0.7678, 0.8734, 0.5489, 0.4189, 0.5612, 0.5714, 0.5000,
        0.7340, 0.6600, 0.3624, 0.5812, 0.5939, 0.6667, 0.4959,
        0.7820, 0.9234, 0.0000, 0.6263, 0.8309, 0.6154, 0.6636,
        0.2857, 0.3288, 0.5333, 0.6875, 0.6311, 0.5699
    ])

    # seed=777
    seed777_f1 = np.array([
        0.7639, 0.8689, 0.5474, 0.3882, 0.5745, 0.5497, 0.5251,
        0.7234, 0.6990, 0.4267, 0.5780, 0.5912, 0.6512, 0.5043,
        0.7852, 0.9300, 0.0000, 0.5941, 0.8427, 0.5926, 0.6577,
        0.3077, 0.3380, 0.6667, 0.6939, 0.6439, 0.6022
    ])

    # seed=999
    seed999_f1 = np.array([
        0.7565, 0.8727, 0.5428, 0.3771, 0.5690, 0.5614, 0.4972,
        0.7147, 0.6667, 0.4000, 0.5960, 0.5963, 0.6667, 0.5124,
        0.8030, 0.9228, 0.0000, 0.6497, 0.8260, 0.6667, 0.6578,
        0.2857, 0.3284, 0.5000, 0.6526, 0.6512, 0.5650
    ])

    # seed=2025
    seed2025_f1 = np.array([
        0.7645, 0.8724, 0.5407, 0.3804, 0.5732, 0.5506, 0.4973,
        0.7104, 0.6863, 0.4196, 0.5846, 0.5897, 0.6087, 0.4833,
        0.7794, 0.9249, 0.0000, 0.6564, 0.8228, 0.6923, 0.6486,
        0.3333, 0.3043, 0.6154, 0.6667, 0.6389, 0.5604
    ])

    # seed = 42
    many_avg_42 = 0.6929
    medium_avg_42 = 0.5394
    few_avg_42 = 0.5566
    group_avg_f1_42 = np.array([many_avg_42, medium_avg_42, few_avg_42])
    row_seed42 = np.concatenate([seed42_f1, group_avg_f1_42])

    # seed = 123
    many_avg_123 = 0.6933
    medium_avg_123 = 0.5277
    few_avg_123 = 0.5361
    group_avg_f1_123 = np.array([many_avg_123, medium_avg_123, few_avg_123])
    row_seed123 = np.concatenate([seed123_f1, group_avg_f1_123])

    # seed = 777
    many_avg_777 = 0.6908
    medium_avg_777 = 0.5380
    few_avg_777 = 0.5542
    group_avg_f1_777 = np.array([many_avg_777, medium_avg_777, few_avg_777])
    row_seed777 = np.concatenate([seed777_f1, group_avg_f1_777])

    # seed = 999
    many_avg_999 = 0.6864
    medium_avg_999 = 0.5359
    few_avg_999 = 0.5375
    group_avg_f1_999 = np.array([many_avg_999, medium_avg_999, few_avg_999])
    row_seed999 = np.concatenate([seed999_f1, group_avg_f1_999])

    # seed = 2025
    many_avg_2025 = 0.6860
    medium_avg_2025 = 0.5288
    few_avg_2025 = 0.5524
    group_avg_f1_2025 = np.array([many_avg_2025, medium_avg_2025, few_avg_2025])
    row_seed2025 = np.concatenate([seed2025_f1, group_avg_f1_2025])
    
    seeds = [42,123,777,999,2025]
    row_seeds = [row_seed42, row_seed123, row_seed777, row_seed999, row_seed2025]
    for i in range(len(seeds)):
        np.save(f"../output/{seeds[i]}_f1_one_row.npy", row_seeds[i])
     

def get_seed_avg_tailed_jac_f1():
    # ckpt_list = [
    #         ("sigfull", 42),
    #         ("sigfull", 123),
    #         ("sigfull", 777),
    #         ("sigfull", 999),
    #         ("sigfull", 2025)
    # ]
    ckpt_list = [
            ("sup", 42),
            ("sup", 123),
            ("sup", 777),
            ("sup", 999),
            ("sup", 2025)
    ]
    # output csv
    out_csv = "./avg_tailed_jac_f1_result.csv"

    _, id2name = get_dicts('goemotions')
    num_classes = len(id2name.items())

    class_avg_jaccard = get_classwise_avg_jaccard()
    jac_median = np.median(class_avg_jaccard)
    high_overlap_cls_idx = class_avg_jaccard >= jac_median
    low_overlap_cls_idx = class_avg_jaccard < jac_median

    rows = []
    for method_name, seed in ckpt_list:
        print(f"\n==== Evaluating {method_name}, seed={seed} ====")
        
        ## read .pt file
        # per_class_f1, many_f1,medium_f1,few_f1 = get_per_class_f1(method_name,seed)
        ## read f1 storage npy file
        f1_loaded = np.load(f"../output/{method_name}_{seed}_f1_one_row.npy")
        per_class_f1 = f1_loaded[:27]
        group_avg_load = f1_loaded[27:]
        many_f1, medium_f1, few_f1 = group_avg_load
        # 标注重叠分组F1
        high_jac_f1 = np.mean(per_class_f1[high_overlap_cls_idx])
        low_jac_f1 = np.mean(per_class_f1[low_overlap_cls_idx])

        row = {
            "method": method_name,
            "seed": seed,
            "many_f1": many_f1,
            "medium_f1": medium_f1,
            "few_f1": few_f1,
            "high_overlap_f1": high_jac_f1,
            "low_overlap_f1": low_jac_f1,
        }

        # 追加每一类单独F1
        for cid in range(num_classes):
            row[f"cls_{cid:02d}_f1"] = per_class_f1[cid]
        rows.append(row)
        print(f"Many: {many_f1:.4f}, Medium: {medium_f1:.4f}, Few: {few_f1:.4f}")
        print(f"HighOverlap: {high_jac_f1:.4f}, LowOverlap: {low_jac_f1:.4f}")

    # ====================== SAVE CSV ======================
    if len(rows) > 0:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved all per-seed results to {out_csv}")

    # ====================== Compute Mean & Std across 5 seeds ======================
    
    df = pd.read_csv(out_csv)
    summary = df.groupby("method").agg(
        many_f1_mean=("many_f1", "mean"),
        many_f1_std=("many_f1", "std"),
        medium_f1_mean=("medium_f1", "mean"),
        medium_f1_std=("medium_f1", "std"),
        few_f1_mean=("few_f1", "mean"),
        few_f1_std=("few_f1", "std"),
        high_overlap_f1_mean=("high_overlap_f1", "mean"),
        high_overlap_f1_std=("high_overlap_f1", "std"),
        low_overlap_f1_mean=("low_overlap_f1", "mean"),
        low_overlap_f1_std=("low_overlap_f1", "std"),
    ).reset_index()

    print("\n==== 5-seed Summary ====")
    print(summary.round(4))
    summary.to_csv(f"./group_f1_summary.csv", index=False)
    print("Summary saved to group_f1_summary.csv")

if __name__ == '__main__':
    # plot_umap()
    # get_classwise_avg_jaccard()
    # get_1seed_overlap_f1()
    # get_1seed_cross_tailed_jac_f1()
    # save_oasi_f1()
    get_seed_avg_tailed_jac_f1()