import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import trange
from tqdm.auto import tqdm
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode='min', verbose=False):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        

    def __call__(self, score):
        score = float(score)
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            elif score - self.best_score > 1.0:
                self.early_stop = True
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                            f'Best: {self.best_score:.5f}' \
                            f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                            f'Best: {self.best_score:.5f}' \
                            f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False
class Contrastive_Loss(): 
    def __init__(self, temperature, batch_size, pos_neg):
        self.temperature = temperature
        self.batch_size = batch_size
        self.pos = pos_neg
        
    def __call__(self, out, do_normalize=True):
        if self.pos:
            if do_normalize:
                out = F.normalize(out, dim=1)
            batch_size = int(out.shape[0]/2)

            # drop_last를 위함
            if batch_size != self.batch_size:
                bs = batch_size
            else:
                bs = self.batch_size

            # out_1:x, out_2:pos
            out_1, out_2 = out.split(bs, dim=0) # (B,D), (B,D)

            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature) # B,B
            mask = (torch.ones_like(sim_matrix) - torch.eye(2*bs, device=sim_matrix.device)).bool()       
            sim_matrix = sim_matrix.masked_select(mask).view(2*bs, -1) # 2B,1

            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0) # 2B,1

            loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

            return loss   
        
        else:
            if do_normalize:
                out = F.normalize(out, dim=1)
            batch_size = int(out.shape[0]/3)

            if batch_size != self.batch_size:
                bs = batch_size
            else:
                bs = self.batch_size
            origin, pos, neg = out.split(bs, dim=0) # (B,D), (B,D), (B,D)

            # anchor vs pos
            sim_matrix_pos = torch.exp(torch.mm(origin, pos.t().contiguous()) / self.temperature)
            # anchor vs neg
            sim_matrix_neg = torch.exp(torch.mm(origin, neg.t().contiguous()) / self.temperature)

            pos_sim = torch.exp(torch.sum(origin * pos, dim=-1) / self.temperature)

            loss = (-torch.log(pos_sim / (sim_matrix_pos.sum(dim=-1) + sim_matrix_neg.sum(dim=-1)))).mean()

            return loss
def set_columns(dataset):
    dataset = pd.DataFrame(
        {"context": dataset["context"], "query": dataset["question"], "title": dataset["title"]}
    )

    return dataset


def load_tokenizer(MODEL_NAME):
    special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def tokenize_colbert(dataset, tokenizer, corpus):

    # for inference
    if corpus == "query":
        preprocessed_data = []
        for query in dataset:
            preprocessed_data.append("[Q] " + query)

        tokenized_query = tokenizer(
            preprocessed_data, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        return tokenized_query

    elif corpus == "doc":
        preprocessed_data = "[D] " + dataset
        tokenized_context = tokenizer(
            preprocessed_data,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return tokenized_context

    elif corpus == "bm25_hard":
        preprocessed_context = []
        for context in dataset:
            preprocessed_context.append("[D] " + context)
        tokenized_context = tokenizer(
            preprocessed_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return tokenized_context
    # for train
    else:
        preprocessed_query = []
        preprocessed_context = []
        for query, context in zip(dataset["query"], dataset["context"]):
            preprocessed_context.append("[D] " + context)
            preprocessed_query.append("[Q] " + query)
        tokenized_query = tokenizer(
            preprocessed_query, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        tokenized_context = tokenizer(
            preprocessed_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return tokenized_context, tokenized_query