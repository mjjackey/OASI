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
from transformers import AdamW, get_linear_schedule_with_warmup,RobertaForSequenceClassification
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


def train(model, train_loader, opt, scheduler, accum_iter, current_epoch, global_step):
    scaler = GradScaler()
    model.train()

    epoch_loss, all_preds, all_labels = 0, [], []
    batch_idx = 0
    ## Get labels from the dataloader
    for input_ids, attention_mask, labels, soft_p_tensor, lbl_set_list in tqdm(train_loader):
        input_ids, attention_mask, labels, soft_p_tensor = input_ids.to(args.device), attention_mask.to(args.device),\
                                                            labels.to(args.device), soft_p_tensor.to(args.device), 

        ## Invoke forward() function,   enabled=False
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
            # logits' shape: [batch_size, num_class]
            lbl_pred_logits, cls_loss, contra_loss, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, \
                                                                 soft_p_tensor=soft_p_tensor,  lbl_set_list=lbl_set_list, current_epoch=current_epoch)
        
            if torch.isnan(loss).any() or torch.any(torch.isinf(loss)):
                print(f">>> detect nan or inf in total loss")
                raise RuntimeError("loss is nan before backward")

             # Gradient accumulation
            loss_backward = loss / accum_iter
            
        batch_loss = loss.item()
        scaler.scale(loss_backward).backward()

        ## Achieve the gradient accumalation
        if (batch_idx+1) % accum_iter == 0:
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
            global_step += 1
            ## Log the metrics to wandb
            wandb.log({"global_step": global_step, "lr": scheduler.get_last_lr()[0]})
   
        epoch_loss += batch_loss
        all_preds.extend(torch.argmax(lbl_pred_logits, dim=-1).tolist())
        all_labels.extend(labels.tolist())

        if batch_idx % 400 ==0:
            print(f"step:{batch_idx}  cls_loss:{cls_loss.item():.4f}  contra_loss:{contra_loss.item():.4f} batch_total_loss:{batch_loss:.4f}")

        batch_idx+=1
    
    train_avg_loss = epoch_loss / batch_idx  # total_loss / len(train_loader)
    
    train_acc = accuracy_score(all_labels, all_preds)
    train_weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Train => loss : {train_avg_loss:.04f}, '
          f'acc : {train_acc:.04f}, '
          f'f1 : {train_weighted_f1:.04f}, '
          f'macro_f1 : {macro_f1:.04f}')

    return train_avg_loss, train_acc, train_weighted_f1, macro_f1, global_step


def eval_or_test(model, loader, mode):
    model.eval()

    all_logits = []
    total_loss,all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels, soft_p_tensor, lbl_set_list in loader:
            input_ids, attention_mask, labels, soft_p_tensor = input_ids.to(args.device), attention_mask.to(args.device), \
                                                            labels.to(args.device), soft_p_tensor.to(args.device)
            logits, _, _, loss = model(input_ids, attention_mask, labels, soft_p_tensor,lbl_set_list)

            batch_loss = loss.item()
            total_loss += batch_loss
            all_preds.extend(torch.argmax(logits, dim=-1).tolist())
            all_labels.extend(labels.tolist())
            all_logits.extend(logits.tolist())

    eval_avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(mode + f' => loss : {eval_avg_loss:.04f}, '
                 f'acc : {acc:.04f}, '
                 f'weighted_f1 : {weighted_f1:.04f}, '
                 f'macro_f1 : {macro_f1:.04f}')

    return eval_avg_loss, acc, weighted_f1, macro_f1


FROM_PRE_TRAINED = False
FROM_PRE_TRAINED_WHOLE = False
GRADIENT_CHECK= True
def main():
    _, idx2label = get_dicts(args.dataset)
    num_classes = len(idx2label.items())
    print("num_classes: ", num_classes)
    class_names = [v for k, v in sorted(idx2label.items(), key=lambda item: item[0])]

    # _tau_ov{args.temperature_ov}
    ckpt_name = f"best_seed{args.seed}_lc{args.lambda_contrast}_tau{args.temperature}_tau_ov{args.tau_ov}.pt" 
    ckpt_path = os.path.join(args.save_dir, ckpt_name)
    if FROM_PRE_TRAINED_WHOLE:
        if not os.path.exists(ckpt_path):
            print(f"model_path {ckpt_path} does not exist, please check the path")
            sys.exit(1)
        
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
    if FROM_PRE_TRAINED:
        # model = RobertaForSequenceClassification.from_pretrained(args.encoder_type, num_labels=num_classes)
        # # Ensure classifier head matches 27 classes even if ckpt config differs.
        # model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
        state_dict = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state_dict,strict=False)

    model.to(args.device)
    # Gradient checkpointing
    if GRADIENT_CHECK:
        # if hasattr(model.roberta, "gradient_checkpointing_enable"):
        #     model.roberta.gradient_checkpointing_enable()
        # else:
        #     print("gradient checkpoint not available for this transformers version")
        model.encoder.gradient_checkpointing_enable()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    opt = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    ## Compute real update steps
    # len(train_loader): the number of batches per epoch, args.accum_iter: optimiser.step() is called every args.accum_iter batches
    update_per_epoch =  (len(train_loader) // args.accum_iter) 
    total_update_steps = update_per_epoch * args.epochs
    warmup_ratio = 0.06
    start_epochs = 0
    ### If observe train loss oscillates, increase `warmup_ratio` to `0.08`.
    if FROM_PRE_TRAINED: 
        warmup_ratio = 0.1
        start_epochs = args.start_epoch
        remaining_epochs = args.epochs - args.start_epoch
        total_update_steps = max(remaining_epochs, 0) * update_per_epoch

    warmup_steps = int(total_update_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
                optimizer=opt,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_update_steps
    )
    
    early_stopping = EarlyStopping(args,patience=args.patience, delta=1e-4, type='macro_f1', path=ckpt_path)

    # for args.seed in [42,123,777]:
    #     for args.lambda_contrast in [0.1,0.05]:
    best_val_macro = 0
    global_step = 0

    if FROM_PRE_TRAINED_WHOLE:
        state_dict = torch.load(ckpt_path, map_location=args.device)
        opt.load_state_dict(state_dict["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        start_epochs = state_dict["epoch"] + 1
        global_step = state_dict["global_step"]
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
    try: 
        for epoch in range(start_epochs,args.epochs):
            print(f'----------- {epoch + 1} -------------')

            train_loss, train_acc, train_f1, train_macro_f1, global_step = train(model, train_loader, opt, scheduler, args.accum_iter, epoch,  global_step)
            eval_loss, eval_acc, eval_f1, eval_macro_f1 = eval_or_test(model, valid_loader, 'Eval')
            if eval_macro_f1 > best_val_macro:
                best_val_macro = eval_macro_f1
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'train_macro_f1': train_macro_f1,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc,
                'eval_f1': eval_f1,
                'eval_macro_f1': eval_macro_f1
            },step=global_step)

            early_stopping(val_loss=eval_loss, val_metric=eval_macro_f1, model=model, \
                            optimiser=opt,scheduler=scheduler,epoch=epoch,global_step=global_step)
            if early_stopping.early_stop:
                print(f"+++ early stop at epoch {epoch + 1 - args.patience} +++")
                break

        print(f'----------- test -------------')
        # model.load_state_dict(ckpt_path, map_location=args.device)
        state_dict = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state_dict,strict=False)
        test_loss, test_acc, test_f1, test_macro_f1 = eval_or_test(model, test_loader, 'Test')
        print('test_acc: ', {test_acc}, 'test_f1: ', {test_f1}, 'test_macro_f1: ', {test_macro_f1})
        wandb.log({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_macro_f1': test_macro_f1,
        },global_step)

        wandb.summary["test_acc"] = test_acc
        wandb.summary["test_weighted_f1"] = test_f1
        wandb.summary["test_macro_f1"] = test_macro_f1
        wandb.summary["best_val_macro_f1"] = best_val_macro
    
    except KeyboardInterrupt:
        print("==== KeyboardInterrupt: User Ctrl+C stopped training ====")
    except Exception as e:
        print(f"==== Training exception caught: {type(e).__name__}: {e}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        print("==== Call wandb.finish() ====")
        wandb.finish()

def wandb_init(args):
    project="goemotion_soft_infonce"
    config={
                "batch_size": args.batch_size,
                "accum_iter": args.accum_iter,
                "effective_batch": args.batch_size * args.accum_iter,
                "lr": args.lr,
                "dataset": args.dataset,
                "encode_type": args.encoder_type,
                "device_id": args.device_id,
                "save_dir": args.save_dir,
                # "start_epoch": args.start_epoch,
            }
    
    name=None
    notes=None
    if FROM_PRE_TRAINED:
        name="resume_from_epoch4_best_ckpt"
        notes="load model weights from epoch4 best ckpt; optimizer & scheduler reinitialized, no opt state saved"
    elif FROM_PRE_TRAINED_WHOLE:
        name="resume_from_epoch_best_ckpt"
        notes="load model weights, optimizer & scheduler from epoch best ckpt"
    wandb.init(project=project, name=name, config=config, notes=notes) # Initialize wandb
    cfg = wandb.config
    for k,v in cfg.items():
        setattr(args, k, v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='goemotions')
    parser.add_argument('--batch_size', type=int, default=4)  # original: 64
    parser.add_argument('--accum_iter', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--encoder_type', type=str, default='roberta-base')
    parser.add_argument('--seed', type=int, default=0)
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

    args = parser.parse_args()
    args.device = torch.device("cuda:" + args.device_id if torch.cuda.is_available() else "cpu")

    wandb_init(args) # Initialize wandb

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main()
