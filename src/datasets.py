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
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# import wandb
import datetime
import csv

all_dataset_list = ['goemotions', 'ED']

#TODO path
EMO_RAW_CSV_PATHS = [
   f'../data/goemotions/goemotions_1.csv',
   f'../data/goemotions/goemotions_2.csv',
   f'../data/goemotions/goemotions_3.csv',
]
EMO_OUT_PATH = '../data/goemotions'
EMOTION_COLS = [
        'admiration','amusement','anger','annoyance','approval','caring','confusion',
        'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
        'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
        'pride','realization','relief','remorse','sadness','surprise','neutral'
        ]
N_EMO = len(EMOTION_COLS)  # 28
TRAIN_TSV = "train.tsv"   # goemotions original split data
DEV_TSV   = "dev.tsv"
TEST_TSV  = "test.tsv"
ANNO_COLS = 1
MULTI_LABEL = False  # whether to keep multi-label samples, if False, only keep single-label samples

def get_dicts(dataset):
    if dataset == 'ED':
        label2idx = {'sad': 0, 'trusting': 1, 'terrified': 2, 'caring': 3, 'disappointed': 4,
                     'faithful': 5, 'joyful': 6, 'jealous': 7, 'disgusted': 8, 'surprised': 9,
                     'ashamed': 10, 'afraid': 11, 'impressed': 12, 'sentimental': 13,
                     'devastated': 14, 'excited': 15, 'anticipating': 16, 'annoyed': 17, 'anxious': 18,
                     'furious': 19, 'content': 20, 'lonely': 21, 'angry': 22, 'confident': 23,
                     'apprehensive': 24, 'guilty': 25, 'embarrassed': 26, 'grateful': 27,
                     'hopeful': 28, 'proud': 29, 'prepared': 30, 'nostalgic': 31}

    elif dataset == 'goemotions':
        label2idx = {'admiration': 0, 'amusement': 1, 'anger': 2,
                     'annoyance': 3, 'approval': 4, 'caring': 5,
                     'confusion': 6, 'curiosity': 7, 'desire': 8,
                     'disappointment': 9, 'disapproval': 10, 'disgust': 11,
                     'embarrassment': 12, 'excitement': 13, 'fear': 14,
                     'gratitude': 15, 'grief': 16, 'joy': 17,
                     'love': 18, 'nervousness': 19, 'optimism': 20,
                     'pride': 21, 'realization': 22, 'relief': 23,
                     'remorse': 24, 'sadness': 25, 'surprise': 26}

    idx2label = {v: k for k, v in label2idx.items()}
    return label2idx, idx2label

def build_annotator_distri(ds_type:str):
    """
    Build annotator distribution for the given dataset type.
    Handle the row raw full dataset, go_emotios_1.csv, go_emotions_2.csv, go_emotions_3.csv
    Args:
        ds_type: dataset type
    Returns:
        dictionary mapping example IDs to their soft targets
    """
    if ds_type == "goemotions":
        # load raw full dataset, go_emotios_1.csv, go_emotions_2.csv, go_emotions_3.csv and concat them
        df_raw = pd.concat([pd.read_csv(p) for p in EMO_RAW_CSV_PATHS], ignore_index=True)
     
        missing = [c for c in EMOTION_COLS if c not in df_raw.columns]
        assert not missing, f"raw csv lack emotion label column: {missing}"

        soft_target_dict = {}
        for cid, group in df_raw.groupby("id"):
            # filter out raters who marked example_very_unclear
            valid = group[group["example_very_unclear"] == False]
            num_raters = len(valid)
            if num_raters == 0:
                continue
            ## One dim series with 28 bool values, true means voted column label 
            df_col_bool = (valid[EMOTION_COLS] != 0).any(axis=0)
            raw_multi_lbl_indice = [idx for idx, flag in enumerate(df_col_bool) if flag]
            # 28 dims vector,  ndarray
            counts = valid[EMOTION_COLS].sum(axis=0).values  
            # print("counts: ", type(counts).__name__)
            # 28 dims annotator-distribution vector, [0~1]^28
            prob_vec = counts / num_raters
            soft_target_dict[cid] = (prob_vec, raw_multi_lbl_indice)
    return soft_target_dict


def load_data_split(path):
    """
    Handle the row .tsv split file, return a pandas DataFrame.
    Args:
        path: path to .tsv file
    Returns:
        pandas DataFrame
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            text, label_str, cid = parts[0], parts[1], parts[2]
            # label_ids = [int(x) for x in label_str.split(",") if x.strip() != ""]
            rows.append({ "id": cid, "text": text, "label_ids": label_str})
    return pd.DataFrame(rows)
            

def merge_and_save(split_df, soft_target_dict, out_path):
    missing = split_df[~split_df["id"].isin(soft_target_dict)].index
    if len(missing) > 0:
        print(f"[WARN] {len(missing)} samplies cannot be found in the raw csv, will be ignored: {out_path}")

    # Filter out the df whose id is the soft_target_dict's key
    df = split_df[split_df["id"].isin(soft_target_dict)].copy()

    # Store soft target as 28 columns
    if ANNO_COLS == 28:
        # shape: [num_samples, 28]
        soft_mat = np.vstack([soft_target_dict[cid] for cid in df["id"]])
        soft_cols = [f"soft_{c}" for c in EMOTION_COLS]
        soft_df = pd.DataFrame(soft_mat, columns=soft_cols)
        handled_df = pd.concat([df.reset_index(drop=True), soft_df], axis=1)
        # label_ids converts to comma split string for storage convenience
        # handled_df["label_ids"] = handled_df["label_ids"].apply(lambda x: ",".join(map(str, x)))
        # quoting=csv.QUOTE_MINIMAL to keep the comma split string in one column
        handled_df.to_csv(out_path, index=False, encoding="utf-8",quoting=csv.QUOTE_MINIMAL)
        print(f"[OK] saved {len(handled_df)} rows -> {out_path}")
    # Store soft target as 1 column comma split string: sof_prob_str
    else: 
        soft_anno_list = [] # contain all row's soft_prob_str
        mult_lbl_list = []
        for cid in df["id"]:
            vec, lbl_indice = soft_target_dict[cid]  # shape (28,) np array
            # ndarray -> comma split string
            vec_str = ",".join([f"{x:.6f}" for x in vec])
            soft_anno_list.append(vec_str)
            # list -> comma split string
            mult_lbl_str = ",".join([str(x) for x in sorted(lbl_indice)])
            mult_lbl_list.append(mult_lbl_str)

        df["soft_prob_str"] = soft_anno_list
        df["raw_label_ids"] = mult_lbl_list
        # quoting=csv.QUOTE_MINIMAL to keep the comma split string in one column
        df.to_csv(out_path, index=False, encoding="utf-8",quoting=csv.QUOTE_MINIMAL)
        handled_df = df
        print(f"[OK] saved {len(df)} rows -> {out_path}")

    return handled_df


def filter_single_lbl_no_neutral(input_csv, output_csv):
    """
    Handle soft.csv
    Args:
        input_csv: Entire csv with soft_prob_str (label_ids comma split, soft_prob_str 28 dimensions vector)
        output_csv: Single label, no neutral label, 27 class samples
    """
    df = pd.read_csv(input_csv)
    print(f"Input Whole Sample Count: {len(df)}")

    keep_rows = []
    for _, row in df.iterrows():
        label_str = str(row["label_ids"])
        label_ids = [int(x.strip()) for x in label_str.split(",") if x.strip() != ""]
        
        # 1. Keep single label
        if len(label_ids) != 1:
            continue
        
        label_id = label_ids[0]
        # 2. Eliminate neutral label
        if label_id == 27:
            continue
        
        soft_p_str = row["soft_prob_str"]
        # string --> [28] dimensions vector
        soft_arr = np.fromstring(soft_p_str, sep=",", dtype=np.float32)
        soft_arr_27 = soft_arr[:27]  
        # [27] dimensions vector --> string
        soft_p_str_27 = ",".join([f"{x:.6f}" for x in soft_arr_27])
        
        keep_rows.append({
            "text": row["text"],
            "label": label_id,         
            "id": row["id"],
            "raw_label_ids":  row["raw_label_ids"],
            "soft_prob_str": soft_p_str_27
        })

    out_df = pd.DataFrame(keep_rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Output single label without neutral samples: {len(out_df)} -> {output_csv}")
    return out_df


def handle_filter():
    for split in ["train", "valid", "test"]:
        in_path = f"../data/goemotions/goemotions_{split}_with_soft.csv"
        out_path = f"../data/goemotions/goemotions_{split}_single_27.csv"
        filter_single_lbl_no_neutral(in_path, out_path)
 

def handle_dataset(type):
    if type == 'goemotions':
        anno_dict = build_annotator_distri(type)
        print(f"There are {len(anno_dict)} samples having annotator-distribution")
        for tsv, name in [(TRAIN_TSV, "train"), (DEV_TSV, "valid"), (TEST_TSV, "test")]:
            split_df = load_data_split(f"{EMO_OUT_PATH}/{tsv}")
            out = f"{EMO_OUT_PATH}/goemotions_{name}_with_soft.csv"
            merge_and_save(split_df, anno_dict, out)


def verify_dataset():
    """
    Verify the dataset by checking the number of samples and labels.
    """
    ds_type = 'goemotions'
    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(f'../data/{ds_type}/{ds_type}_{split}_single_27.csv')
        print(f"{ds_type} {split} sample count: {len(df)}")
        print(f"{ds_type} {split} unique labels: {sorted(df['label'].unique())}")
        dims = df["soft_prob_str"].apply(lambda s: len(np.fromstring(s, sep=",", dtype=np.float32)))
        print(f"soft_p_str dims: {dims.unique()}")   # [27]
        print(f"dataset's shape: {df.shape}, columns: {df.columns.tolist()}")


class emotion_DataSet(Dataset):
    def __init__(self, mode, tkr, ds_type):
        """
        mode: 'train', 'valid', 'test'
        tkr: tokenizer instance
        type: 'ED' or 'go_emotion'
        """
        super().__init__()
        assert mode in ['train', 'valid', 'test']
        df = pd.read_csv(f'../data/{ds_type}/{ds_type}_{mode}_single_27.csv')
        df = df[df["label"] != -1].reset_index(drop=True)

        print("label min:",df["label"].min(),"label max:",df["label"].max())
        print("unique labels:",sorted(df["label"].unique()))

        self.df = df
        self.tkr = tkr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        if MULTI_LABEL:
            # hard multi‑hot label_ids
            label = [int(x) for x in str(row["label_ids"]).split(",") if x!=""]
        else:
            # single label
            label=row["label"]
        
        lbl_ids_str = row["raw_label_ids"]
        lbl_ids = [int(x.strip()) for x in lbl_ids_str.split(",")]
        row_lbl_set = set(lbl_ids)

        # deserialize annotator‑distribution soft vector
        soft_p_str = row["soft_prob_str"]
        soft_p_np = np.fromstring(soft_p_str, sep=",", dtype=np.float32) # shape (num_classes,)
        soft_p_tensor = torch.from_numpy(soft_p_np)

        return (text,label,soft_p_tensor,row_lbl_set)

    def collate(self, batch):
        """
        Args:
            batch: list of tuples (text, label, soft_p_tensor, s_i)
        Returns: 
            input_ids: token id, shape: [batch_size, seq_len], 
            attention_mask: 1 or 0, shape: [batch_size, seq_len],
            label_tensor: shape: [B] for single label, CrossEntropyLoss; [B, uncertain_dim] for multi_label, BCEWithLogitsLoss
            soft_p_tensor: [B, num_classes] annotator‑distribution
            row_lbl_set_list: a list with size B, containing row multi-label set
        """
        texts = [t for t,_, _, _ in batch]
        encode = self.tkr(texts, padding='longest', truncation=True, max_length=200, return_tensors='pt')
 
        labels = [l for _, l, _, _ in batch]
        label_tensor = torch.tensor(labels, dtype=torch.long)

        # a list of tensors, where one tensor's shape is [num_classes]
        soft_p_tensors = [s for _, _, s,_ in batch]  
        # Deserialise soft_p_tensors -> tensor [B,num_classes]
        soft_p_tensor = torch.stack(soft_p_tensors,dim=0)  # shape: [B, num_classes]

        row_lbl_set_list = [ls for _, _, _, ls in batch]

        return encode['input_ids'], encode['attention_mask'], label_tensor, soft_p_tensor, row_lbl_set_list

if __name__=="__main__":
    # handle_dataset('goemotions')
    # handle_filter()
    verify_dataset()
