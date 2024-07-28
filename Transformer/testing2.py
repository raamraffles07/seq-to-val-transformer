## d_model --> dimensional size of every vector throughout the encoder architecture. 
## num_heads --> number of heads the query, key and value vectors are split into in multi-headed attention.
## drop_prob --> drop probability *throughout* the encoder layers --> regularisation, 
## prevents model from just memorising specific patterns
## batch_size --> number of amino acid sequences in each training batch
## max_sequence_length --> maximum number of amino acids in each amino acid sequence 
## ffn_hidden --> dimensionality of the feed forward neural network in the encoder layer,
## to allow the different attention heads to communicate with one another. 
## num_layers --> number of encoder layers in the architecture 


## Adjustable Hyperparameters:
##d_model = 1024
##num_layers = 12

## More research needed to find optimum max. seq length:
##max_sequence_length = 1500

##num_heads = 8
##drop_prob = 0.10
##batch_size = 20
##ffn_hidden = 2048
## encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
## transformer --> sequence embedding + positional encoding --> x --> encoder (Done) --> linear output

import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F

BIOPHYSICS = {  'A': [1.8,  0,  6.0,  89.1, 0, 0,  75.8,  76.1,    1.9],
                'z': [0, 0, 0, 0, 0, 0, 0, 0, 0],
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

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask):
    
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        # Broadcasting add. So just the last N dimensions need to match
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    #print("\n" + "scaled dot product sucessful")
    return values, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, max_sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        #print("mh-attention mechanism sucessful")
        return out
    
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y  + self.beta
        #print("Layer Normalisation successful" + "\n")
        return out

  
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        #print("FFN successful")
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads)
        # Specific instance of the Multi-Headed attention class, with d_model specified by the tranformer's d_model
        # and num_heads specified by the transformer's num_heads, initialises d_model, num_heads, head_dim, automatically 
        # creates 2 linear layers, one to map x to query, key and value vectors, and one to map with a x - x vector.
        # Note that the specific instance itself is self.attention

        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        
        residual_x = x
        x = self.attention(x, mask = None)
        # self.attention(x) is the same as self.attention.forward(x)

        # x = self.se(x, st, et)

        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)

        return x

class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                     for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x

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
        #print("\n" + "amino acid embedding is successful")
        return x

class finalconv(nn.Module):
    def __init__(self, max_sl, d_model):
        super().__init__()
        self.dime = (max_sl * d_model)
        self.f_linear = nn.Linear(self.dime, 1)
    
    def forward(self, x):
        a, b, c = x.size()
        x = x.view(a, -1)
        x = self.f_linear(x)
        x = F.relu(x)
        
        return x


    
class Transformer(nn.Module):
    def __init__(self, start_token, end_token, padding_token, biophysics, max_sl, d_model, dropout, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.d_model = d_model
        self.a_embedding = aa_embedding(start_token, end_token, padding_token, biophysics, max_sl, d_model, dropout)
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                     for _ in range(num_layers)])
        self.flayer = finalconv(max_sl, d_model)

    
    def forward(self, x):
        x = self.a_embedding(x)
        x = self.layers(x)
        x = self.flayer(x)
        print(x)
        return x


a = Transformer("Y", "z", "E", BIOPHYSICS, 6, 16, 0.1, 2048, 8, 0.1, 6)

b = ["ACDG", "GPNM"]

a.forward(b)

