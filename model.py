import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    AutoConfig
)


class ColbertModel(nn.Module):
    def __init__(self, config):
        super(ColbertModel, self).__init__()

        # BertModel 사용
        self.similarity_metric = "cosine"
        self.dim = 128
        self.plm = AutoModel.from_config(config)
        self.plm.init_weights()
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)

    def forward(self,q_inputs, p_inputs, n_inputs=None):
        Q = self.query(**q_inputs)
        print('Q.shape',Q.shape)
        D = self.doc(**p_inputs)
        if n_inputs:
            N = self.doc(**n_inputs)
            return self.get_score(Q, D, N)
        else:
            return self.get_score(Q, D)

    def query(self, input_ids, attention_mask):
        Q = self.plm(input_ids, attention_mask=attention_mask)['last_hidden_state']
        Q = self.linear(Q)
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask):
        D = self.plm(input_ids, attention_mask=attention_mask)['last_hidden_state']
        D = self.linear(D)
        return torch.nn.functional.normalize(D, p=2, dim=2)

    def get_score(self, Q, D, N=None, eval=False):
        # hard negative N은 train에만 쓰임.
        if eval:
            if self.similarity_metric == "cosine":
                final_score = torch.tensor([])
                for D_batch in tqdm(D):
                    D_batch = np.array(D_batch)
                    D_batch = torch.Tensor(D_batch).squeeze()
                    print(D_batch.shape)
                    p_seqeunce_output = D_batch.transpose(
                        1, 2
                    )  # (batch_size,hidden_size,p_sequence_length)
                    q_sequence_output = Q.view(
                        Q.shape[0], 1, -1, self.dim
                    )  # (batch_size, 1, q_sequence_length, hidden_size)
                    dot_prod = torch.matmul(
                        q_sequence_output, p_seqeunce_output
                    )  # (batch_size,batch_size, q_sequence_length, p_seqence_length)
                    max_dot_prod_score = torch.max(dot_prod, dim=3)[
                        0
                    ]  # (batch_size,batch_size,q_sequnce_length)
                    score = torch.sum(max_dot_prod_score, dim=2)  # (batch_size,batch_size)
                    final_score = torch.cat([final_score, score], dim=1)
                print(final_score.size())
                return final_score
        else:
            p_seqeunce_output = D.transpose(
                1, 2
            )  # (batch_size, hidden_size, p_sequence_length)

            q_sequence_output = Q.view(
                Q.shape[0], 1, -1, self.dim
            )  # (batch_size, 1, q_sequence_length, hidden_size)

            dot_prod = torch.matmul(
                q_sequence_output, p_seqeunce_output
            )  # (batch_size,batch_size + 1, q_sequence_length, p_seqence_length)

            max_dot_prod_score = torch.max(dot_prod, dim=3)[0]  # (batch_size,batch_size + 1,q_sequnce_length)

            final_score = torch.sum(max_dot_prod_score, dim=2)  # (batch_size,batch_size + 1)

            return final_score

if __name__=='__main__':
    config = AutoConfig.from_pretrained('monologg/koelectra-base-v3-discriminator')
    model = ColbertModel(config)
    print(model)