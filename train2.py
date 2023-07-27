from utils import Contrastive_Loss,EarlyStopping
from model import ColbertModel
from transformers import AutoModel,AutoTokenizer, AutoConfig
from omegaconf import OmegaConf
import wandb
from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import _LRScheduler,CosineAnnealingWarmRestarts
from torch import optim
from torch.nn.utils import clip_grad_norm_

from tqdm.auto import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import random
import logging
import os
import torch.nn.functional as F
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2"
from datasets import Dataset,load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
# run ddp
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl',rank=rank, world_size= world_size)

class Datacollatewithpadding:
    def __init__(self,cfg):
        self.cfg = cfg

    def __call__(self, batch):
        claim_input_ids = [item['input_ids'] for item in batch]
        claim_attention_masks = [item['attention_mask'] for item in batch]
        pos_input_ids = [item['pos_input_ids'] for item in batch]
        pos_attention_masks = [item['pos_attention_mask'] for item in batch]
        # 각 텐서의 최대 길이를 계산합니다
        # max_length = max(
        #    max(len(ids) for ids in claim_input_ids),
        #    max(len(ids) for ids in pos_input_ids),
        #    max(len(ids) for ids in neg_input_ids)
        # )
        # 동적할당을 하려구 하니까 이거 안될듯
        claim_padded_input_ids = pad_sequence(claim_input_ids,batch_first=True)
        claim_padded_attention_masks = pad_sequence(claim_attention_masks, batch_first=True)
        pos_padded_input_ids = pad_sequence(pos_input_ids, batch_first=True)
        pos_padded_attention_masks = pad_sequence(pos_attention_masks, batch_first=True)
        # claim_padded_input_ids = torch.LongTensor(claim_input_ids)
        # claim_padded_attention_masks = torch.LongTensor(claim_attention_masks)
        # pos_padded_input_ids = torch.LongTensor(pos_input_ids)
        # pos_padded_attention_masks = torch.LongTensor(pos_attention_masks)
        # print('claim_padded_input_ids',claim_padded_input_ids.shape)

        return {
            'input_ids': claim_padded_input_ids,
            'attention_mask': claim_padded_attention_masks,
            'pos_input_ids': pos_padded_input_ids,
            'pos_attention_mask': pos_padded_attention_masks,
        }

def preprocess_function(example):
    # row 단위 전처리를 어떻게 할 것인지
    global tokenizer,cfg
    origin = tokenizer(example['question'],
                                    add_special_tokens=True,
                                    truncation=True,
                                    padding = True,
                                    return_tensors="pt",
                                    # max_length=512
                                    )
    pos = tokenizer(example['context'],
                                    add_special_tokens=True,
                                    truncation=True,
                                    padding = True,
                                    return_tensors="pt",
                                    # max_length=512
                                    )
    # batch를 완료 했고
    # 여기서 data_collater로 배치단위 패딩을 주고 싶어
    return {'input_ids' : origin['input_ids'],'attention_mask' : origin['attention_mask'], 'pos_input_ids' : pos['input_ids'], 'pos_attention_mask' : pos['attention_mask']}
            


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)               
    random.seed(seed)
    print('lock_all_seed')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(rank,world_size,cfg):
    global tokenizer

    ## Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    ## Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    # add special token
    special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
    tokenizer.add_special_tokens(special_tokens)
    model_config = AutoConfig.from_pretrained(cfg.model.model_name)
    
    model = ColbertModel(model_config)
    model.plm.resize_token_embeddings(len(tokenizer))


    print('Count of using GPUs:', torch.cuda.device_count())


    ## load dataset
    all_data = load_from_disk(cfg.data.train_data)

    # train_dev split, stratify 옵션으로 데이터 불균형 해결!
    # train_data, dev_data = train_test_split(train_data, test_size=cfg.data.train_test_split_ratio, random_state=cfg.train.seed)
    train_data = all_data['train']
    dev_data = all_data['validation']

    print(all_data)
    del all_data
    ## make dataset for pytorch

    data_collator = Datacollatewithpadding(cfg)

    ddp_setup(rank, world_size)
    FC_train_dataset = train_data.map(preprocess_function,batched=True,batch_size=32, num_proc=4,remove_columns=['question','context'])
    FC_dev_dataset = dev_data.map(preprocess_function,batched=True,batch_size=32, num_proc=4,remove_columns=['question','context'])
    FC_train_dataset.set_format(type="torch",device='cpu', columns=['input_ids', 'attention_mask', 'pos_input_ids', 'pos_attention_mask'])
    FC_dev_dataset.set_format(type="torch",device='cpu', columns=['input_ids', 'attention_mask', 'pos_input_ids', 'pos_attention_mask'])
    print('FC_train_dataset',FC_train_dataset['input_ids'][0].shape)

    FC_train_dataloader = DataLoader(FC_train_dataset,batch_size=cfg.train.batch_size,shuffle=False,drop_last=False, collate_fn = data_collator,sampler= DistributedSampler(FC_train_dataset))
    FC_dev_dataloader = DataLoader(FC_dev_dataset,batch_size=cfg.train.batch_size,shuffle=False,drop_last=False, collate_fn = data_collator,sampler = DistributedSampler(FC_dev_dataset))
    

    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.plm.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.train.weight_decay,
        },
        {
            "params": [
                p for n, p in model.linear.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.train.weight_decay,
            "lr" : cfg.train.second_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            },
        ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=cfg.train.lr,weight_decay=0.01,eps = 1e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler.T_0, T_mult=cfg.scheduler.T_mult, eta_min=cfg.scheduler.eta_min)
    # early stopping
    es = EarlyStopping(patience=50, delta=0.0, mode='min', verbose=False)
    model.to(rank)
    model = DDP(model,device_ids=[rank],find_unused_parameters=True)
    if rank==1:
        wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name + str(rank))
        wandb.config.update(OmegaConf.to_container(cfg))
        wandb.watch(model)

    def train_one_epoch(epoch,tbar1,prefix,args):
        min_loss = 5.
        for idx,batch in enumerate(tbar1):
            optimizer.zero_grad()
            # output = model(**batch)
            # 배치 입력을 각각 주고 
            # output 3개 만들어서 
            # loss식 먹인다.
            batch  = {k : v.to(rank) for k, v in batch.items()}
            q_inputs = {
                "input_ids": batch['input_ids'].to(rank),
                "attention_mask": batch['attention_mask'].to(rank)
            }

            p_inputs = {
                "input_ids": batch['pos_input_ids'].to(rank),
                "attention_mask": batch['pos_attention_mask'].to(rank)
            }
            outputs = model(q_inputs,p_inputs) # B, B+N(bm_neg)
            # targets = torch.zeros(outputs.shape[0]).long().to(rank)
            targets = torch.arange(0,outputs.shape[0]).long().to(rank)

            sim_scores = F.log_softmax(outputs, dim=1)

            # loss = F.cross_entropy(outputs, targets)
            loss = F.nll_loss(sim_scores, targets)
            
            del outputs, targets, p_inputs, q_inputs

            # clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.update(loss.item(), cfg.train.batch_size)
            tb_x = epoch * len(tbar1) + idx + 1
            tbar1.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)
            if rank==1:
                wandb.log({f'{prefix}/loss':losses.avg,f'{prefix}/step' :tb_x,'learning_rate' :optimizer.param_groups[0]['lr']})
            if idx % args.train.logging_step == args.train.logging_step-1:
                es(losses.avg)
                if es.early_stop:
                    break
            del batch
            del loss
        return losses.avg
    best_vloss = 10000

    model.zero_grad()
    torch.cuda.empty_cache()
    for epoch in range(cfg.train.epoch):
        losses = AverageMeter()
        torch.cuda.empty_cache()
        print('EPOCH {}:'.format(epoch + 1))
        tbar1 = tqdm(FC_train_dataloader)
        # Make sure gradient tracking is on, and do a pass over the data    
        model.train(True)
        avg_loss = train_one_epoch(epoch, tbar1,'train',cfg)
        

        # We don't need gradients on to do reporting
        # model.train(False)
        model.eval()
        valid_loss = []
        tbar2 =tqdm(FC_dev_dataloader)
        with torch.no_grad():
            losses = AverageMeter()
            for i, batch in enumerate(tbar2):
                batch  = {k : v.to(rank) for k, v in batch.items()}
                
                q_inputs = {
                    "input_ids": batch['input_ids'].to(rank),
                    "attention_mask": batch['attention_mask'].to(rank)
                }

                p_inputs = {
                    "input_ids": batch['pos_input_ids'].to(rank),
                    "attention_mask": batch['pos_attention_mask'].to(rank)
                }

                outputs = model(q_inputs,p_inputs)
                targets = torch.arange(0,outputs.shape[0]).long().to(rank)
                
                sim_scores = F.log_softmax(outputs, dim=1)

                vloss = F.nll_loss(sim_scores, targets)

                del outputs, targets,p_inputs, q_inputs
                valid_loss.append(vloss.item())
                losses.update(vloss.item(), cfg.train.batch_size)
                tbar2.set_description("valid_loss: {0:.6f}".format(losses.avg), refresh=True)
                del batch
                del vloss
            avg_vloss = sum(valid_loss) / len(valid_loss)

        if rank==1:
            wandb.log({'train/avg_loss' : avg_loss,'val/avg_loss' : avg_vloss,'train/epoch' :epoch + 1})
        

        # Log the running loss averaged per batch
        # for both training and validation

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss and cfg.model.save:
            best_vloss = avg_vloss
            # ratio
            model_name = f'{cfg.wandb.exp_name}_E_{epoch+1}_loss{round(best_vloss,5)}.pt'
            # torch.save(model.state_dict(), cfg.model.saved_model + model_name)
            # if DDP
            if rank ==1:
                torch.save(model.module.state_dict(), cfg.data.output_file + model_name)
    destroy_process_group()
    del model


if __name__ == '__main__':
    torch.cuda.empty_cache()

    ## parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    ## set seed
    seed_everything(cfg.train.seed)

    world_size = torch.cuda.device_count()
    mp.spawn(train,args = (world_size,cfg),nprocs=world_size)