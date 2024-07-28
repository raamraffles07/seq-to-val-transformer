import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BIOPHYSICS = {  'A': [1.8,  0,  6.0,  89.1, 0, 0,  75.8,  76.1,    1.9],
                'C': [2.5,  0,  5.1, 121.2, 0, 0, 115.4,  67.9,   -1.2],
                'D': [-3.5, -1,  2.8, 133.1, 0, 1, 130.3,  71.8, -107.3],
                'E': [-3.5, -1,  3.2, 147.1, 0, 1, 161.8,  68.1, -107.3],
                'F': [2.8,  0,  5.5, 165.2, 1, 0, 209.4,  66.0,   -0.8],
                'G': [-0.4,  0,  6.0,  75.1, 0, 0,   0.0, 115.0,    0.0],
                'H': [-3.2,  1,  7.6, 155.2, 0, 1, 180.8,  67.5,  -52.7],  # Avg of HIP and HIE
                'I': [4.5,  0,  6.0, 131.2, 0, 0, 172.7,  60.3,    2.2],
                'K': [-3.9,  1,  9.7, 146.2, 0, 1, 205.9,  68.7, -100.9],
                'L': [3.8,  0,  6.0, 131.2, 0, 0, 172.0,  64.5,    2.3],
                'M': [1.9,  0,  5.7, 149.2, 0, 0, 184.8,  67.8,   -1.4],
                'N': [-3.5,  0,  5.4, 132.1, 0, 1, 142.7,  66.8,   -9.7],
                'P': [-1.6,  0,  6.3, 115.1, 0, 0, 134.3,  55.8,    2.0],
                'Q': [-3.5,  0,  5.7, 146.2, 0, 1, 173.3,  66.6,   -9.4],
                'R': [-4.5,  1, 10.8, 174.2, 0, 1, 236.5,  66.7, -100.9],
                'S': [-0.8,  0,  5.7, 105.1, 0, 1,  95.9,  72.9,   -5.1],
                'T': [-0.7,  0,  5.6, 119.1, 0, 1, 130.9,  64.1,   -5.0],
                'V': [4.2,  0,  6.0, 117.1, 0, 0, 143.1,  61.7,    2.0],
                'W': [-0.9,  0,  5.9, 204.2, 1, 1, 254.6,  64.3,   -5.9],
                'Y': [-1.3,  0,  5.7, 181.2, 1, 1, 222.5,  71.9,   -6.1]
                }

a = ["ACD", "GPNM"]

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    
class aa_embedding(nn.Module):

    def __init__(self, start_token, end_token, padding_token, biophysics, max_sl, d_model, dropout):
        super().__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.biophysics = biophysics
        self.max_sl = max_sl
        self.d_model = d_model
        self.dropout = nn.Dropout(p = dropout)
        self.dimensionality = len(next(iter(biophysics.values())))
        self.position_encoder = PositionalEncoding(d_model, self.max_sl)
        self.linearx = nn.Linear(self.dimensionality, self.d_model)
    
    def batch_tokenize(self, batch):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.biophysics[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.biophysics[self.start_token])
            if end_token:
                sentence_word_indicies.append(self.biophysics[self.end_token])
            for _ in range(len(sentence_word_indicies), self.max_sl):
                sentence_word_indicies.append(self.biophysics[self.padding_token])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], self.start_token, self.end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x):
        x = self.batch_tokenize(x)
        x = self.linearx(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        print(x.size())
        return x



pewpewpew = aa_embedding("A", "C", "D", BIOPHYSICS, 6, 12, 0.1)

pewpewpew.forward(a)