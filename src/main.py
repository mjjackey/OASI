import argparse
import gc
import sys, os
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
import wandb
import datetime
from datasets import *
from models import CrossEncoder
from utils import EarlyStopping
# import plotly.express as px
from sklearn.manifold import TSNE
from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings("ignore")


def train(model, train_loader, opt, scheduler, accum_iter, current_epoch):
    scaler = GradScaler()
    model.train()

    total_loss, all_preds, all_labels = 0, [], []
    step = 0
    ## Get labels from the dataloader
    for input_ids, attention_mask, labels, soft_p_tensor in tqdm(train_loader):
        input_ids, attention_mask, labels, soft_p_tensor = input_ids.to(args.device), attention_mask.to(args.device),\
                                                            labels.to(args.device), soft_p_tensor.to(args.device)

        ## Invoke forward() function,   enabled=False
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
            # logits' shape: [batch_size, num_class]
            lbl_pred_logits, cls_loss, contra_loss, loss = model(input_ids, attention_mask, labels, soft_p_tensor, current_epoch)
        
            if torch.isnan(loss).any():
                print(f">>> detect nan in total loss")
                raise RuntimeError("loss is nan before backward")

        loss = loss / accum_iter
        scaler.scale(loss).backward()

        ## Achieve the gradient accumalation
        if (step+1) % accum_iter == 0:
            # Rescale the gradients and clip gradients to avoid gradient explosion
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update the weights
            scaler.step(opt)
            scaler.update() 
            # Reset the gradients to zero for the next iteration
            opt.zero_grad()
            # only step scheduler when weights updated
            scheduler.step() 
   
        total_loss += loss.item()
        all_preds.extend(torch.argmax(lbl_pred_logits, dim=-1).tolist())
        all_labels.extend(labels.tolist())

        if step % 400 ==0:
            print(f"step:{step}  cls_loss:{cls_loss.item():.4f}  contra_loss:{contra_loss.item():.4f} total_loss:{total_loss:.4f}")

        step+=1
    
    train_acc = accuracy_score(all_labels, all_preds)
    train_weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Train => loss : {total_loss / len(train_loader):.04f}, '
          f'acc : {train_acc:.04f}, '
          f'f1 : {train_weighted_f1:.04f}, '
          f'macro_f1 : {macro_f1:.04f}')

    return total_loss, train_acc, train_weighted_f1, macro_f1


def eval_or_test(model, loader, mode):
    model.eval()

    all_logits = []
    total_loss,all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels, soft_p_tensor in loader:
            input_ids, attention_mask, labels, soft_p_tensor = input_ids.to(args.device), attention_mask.to(args.device), \
                                                            labels.to(args.device), soft_p_tensor.to(args.device)
            logits, _, _, loss = model(input_ids, attention_mask, labels, soft_p_tensor)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=-1).tolist())
            all_labels.extend(labels.tolist())
            all_logits.extend(logits.tolist())

    total_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    marco_f1 = f1_score(all_labels, all_preds, average='macro')
    print(mode + f' => loss : {total_loss:.04f}, '
                 f'acc : {acc:.04f}, '
                 f'weighted_f1 : {weighted_f1:.04f}, '
                 f'marco_f1 : {marco_f1:.04f}')

    return total_loss, acc, weighted_f1, marco_f1


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def main():
    label2idx, idx2label = get_dicts(args.dataset)
    num_classes = len(idx2label.items())
    print("num_classes: ", num_classes)
    class_names = [v for k, v in sorted(idx2label.items(), key=lambda item: item[0])]

    args.encode_type = "roberta-base"

    model_path = os.path.abspath(args.encoder_type)
    print(f"model_path = {model_path}")
    print(f"exists? {os.path.exists(model_path)}")
    print(f"is dir? {os.path.isdir(model_path)}")

    tkr = AutoTokenizer.from_pretrained("roberta-base")
    special_tokens_dict = {'additional_special_tokens': ["<#>"]}
    tkr.add_special_tokens(special_tokens_dict)

    trainset = emotion_DataSet('train', tkr, args.dataset)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=trainset.collate)
    validset = emotion_DataSet('valid', tkr, args.dataset)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, collate_fn=trainset.collate)
    testset = emotion_DataSet('test', tkr, args.dataset)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=trainset.collate)

    ## Construct the model
    model = CrossEncoder(args, num_classes, tkr)
    model.to(args.device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    ## Compute real update steps
    total_update_steps = (len(train_loader) // args.accum_iter) * args.epochs
    ### If observe train loss oscillates, increase `warmup_ratio` to `0.08`.
    warmup_ratio = 0.06
    warmup_steps = int(total_update_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
                optimizer=opt,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_update_steps
    )
    early_stopping = EarlyStopping(args, patience=args.patience, delta=1e-4, type='macro_f1', path='../output/mlp_' + args.device_id +'_'+ args.dataset + '.pt')

    # for args.seed in [42,123,777]:
    #     for args.lambda_contrast in [0.1,0.05]:
    best_val_macro = 0
    # ========== WandB init ==========
    # run = wandb.init(
    #     project="goemotion_soft_infonce",
    #     name=f"lambda_{args.lambda_contrast}_tau_{args.temperature}_seed_{args.seed}",
    #     config={
    #         # "seed": args.seed,
    #         "batch_size": args.batch_size,
    #         "accum_iter": args.accum_iter,
    #         "effective_batch": args.batch_size * args.accum_iter,
    #         # "lambda_contrast": args.lambda_contrast,
    #         # "temperature": args.temperature,
    #         "lr": args.lr,
    #         "model_type": args.encoder_type,
    #         "dataset": "goemotions_single_27"
    #     },
    #     reinit=True
    # )
    for epoch in range(args.epochs):
        print(f'----------- {epoch + 1} -------------')

        train_loss, train_acc, train_f1, train_macro_f1 = train(model, train_loader, opt, scheduler, args.accum_iter, epoch)
        eval_loss, eval_acc, eval_f1, eval_macro_f1 = eval_or_test(model, valid_loader, 'Eval')
        if eval_macro_f1 > best_val_macro:
            best_val_macro = eval_macro_f1
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_macro_f1': train_macro_f1,
            'eval_acc': eval_acc,
            'eval_f1': eval_f1,
            'eval_macro_f1': eval_macro_f1
        })

        early_stopping(val_loss=eval_loss, val_metric=eval_macro_f1, model=model)
        if early_stopping.early_stop:
            print(f"+++ early stop at epoch {epoch + 1 - args.patience} +++")
            break

    # model.load_state_dict(torch.load('../mlp' + args.device_id + args.dataset + '.pt', map_location=args.device))
    print(f'----------- test -------------')
    ckpt_name = f"best_seed{args.seed}_lc{args.lambda_contrast}_tau{args.temperature}.pt"
    model.load_state_dict(torch.load(os.path.join(args.save_dir, ckpt_name), map_location=args.device))
    test_loss, test_acc, test_f1, test_macro_f1 = eval_or_test(model, test_loader, 'Test')
    print('test_acc: ', {test_acc}, 'test_f1: ', {test_f1}, 'test_marco_f1: ', {test_macro_f1})
    wandb.log({
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_marco_f1': test_macro_f1,
    })

    wandb.summary["test_acc"] = test_acc
    wandb.summary["test_weighted_f1"] = test_f1
    wandb.summary["test_macro_f1"] = test_macro_f1
    wandb.summary["best_val_macro_f1"] = best_val_macro

    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='goemotions')
    parser.add_argument('--batch_size', type=int, default=4)  # original: 64
    parser.add_argument('--accum_iter', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--encoder_type', type=str, default='roberta-base')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--weight_decay', type=float, default=0.01) # original: 0.001
    parser.add_argument('--patience', type=int, default=4) # 2  # original: 4
    parser.add_argument('--alpha', type=float, default=0.05) #0.05, 0.1, 0.2 original: 0.4
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    # parser.add_argument('--temperature', type=float, default=0.2) #0.15,0.2,0.3 original: 0.5
    # parser.add_argument('--base_temperature', type=float, default=0.07)
    parser.add_argument('--temperature', type=float, default=0.07) # 0.05~0.1
    parser.add_argument('--lambda_contrast', type=float, default=0.1) # 0.1~1.0
    parser.add_argument('--device_id', type=str, default="0")
    parser.add_argument('--save_dir', type=str, default='../output')

    args = parser.parse_args()
    args.device = torch.device("cuda:" + args.device_id if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        project="goemotion_soft_infonce",
        name=None,
        config={
            "batch_size": args.batch_size,
            "accum_iter": args.accum_iter,
            "effective_batch": args.batch_size * args.accum_iter,
            "lr": args.lr,
            "model_type": args.encoder_type,
            "dataset": args.dataset,
            "encode_type": args.encoder_type,
            "device_id": args.device_id,
            "save_dir": args.save_dir,
        },
    )

    cfg = wandb.config
    for k,v in cfg.items():
        setattr(args, k, v)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main()

    # sweep_configuration = {
    #     'method': 'bayes',
    # }
    # wandb.agent(sweep_id, function=main)