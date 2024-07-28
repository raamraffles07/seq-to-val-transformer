import math
import torch
import torch.nn as nn

class PositionalEncoding1(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding1, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

def compare_positional_encodings(d_model, max_seq_length):
    pe1 = PositionalEncoding1(d_model, max_seq_length)
    pe2 = PositionalEncoding2(d_model, max_seq_length)
    
    # Create a dummy input tensor
    dummy_input = torch.zeros((1, max_seq_length, d_model))
    
    # Get the positional encodings from both implementations
    pos_encoding1 = pe1(dummy_input)
    pos_encoding2 = pe2()
    
    # Since pos_encoding2 does not add positional encoding to input,
    # we need to add it to the dummy input for fair comparison
    pos_encoding2 = dummy_input + pos_encoding2.unsqueeze(0)
    
    # Check if the outputs are the same
    are_equal = torch.allclose(pos_encoding1, pos_encoding2, atol=1e-6)
    
    return are_equal, pos_encoding1, pos_encoding2

# Parameters
d_model = 10
max_seq_length = 50

# Compare
are_equal, pos_encoding1, pos_encoding2 = compare_positional_encodings(d_model, max_seq_length)
print("Are the positional encodings equal? ", are_equal)
if not are_equal:
    print("Positional Encoding 1: ", pos_encoding1)
    print("Positional Encoding 2: ", pos_encoding2)