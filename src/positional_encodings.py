import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        
        # 1. Generate the matrix
        pe = self.create_encoding_matrix()
        
        # 2. Save it to the class state so it moves to GPU automatically
        self.register_buffer('pe', pe)
    
    
    def create_encoding_matrix(self):
        even_nums = torch.arange(0,self.d_model,2)
        
        denominator = torch.pow(10000,even_nums/self.d_model)
        
        
        positions = torch.arange(self.max_len).unsqueeze(1)
        
        angles = positions/denominator
        
        
        # 1. Initialize the complete blank PE matrix
        pe = torch.zeros(self.max_len, self.d_model)

        # 2. Slice by even and odd dimensions
        # [:, 0::2] selects all rows, and every 2nd column starting at 0 (0, 2, 4...)
        #pe_even = pe[:, 0::2] 

        # [:, 1::2] selects all rows, and every 2nd column starting at 1 (1, 3, 5...)
        #pe_odd = pe[:, 1::2]  
        
        # pe_even =  pe[:, 0::2] + torch.sin(angles)
        
        # pe_odd = pe[:, 1::2]  + torch.cos(angles)
        
        pe[:, 0::2] = torch.sin(angles)
        
        pe[:,1::2] = torch.cos(angles)
        
        
        return pe
        
        
        
                
        
        
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        
        # Grab the first seq_len rows from our pre-calculated notebook
        # Syntax: [rows, columns] -> [:seq_len, :] means "Rows 0 to seq_len, all columns"
        positions = self.pe[:seq_len, :]
        
        # Add them together (PyTorch automatically broadcasts across the batch dimension)
        return x + positions
    
    
    
    
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        # We only need the positional dictionary here
        self.pos_emb = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        
        # 1. Generate the ticket on the correct hardware device
        positions_ticket = torch.arange(seq_len, device=x.device)
        
        # 2. Hand the ticket to the dictionary to get the rows
        positions = self.pos_emb(positions_ticket)
        
        # 3. Add them together
        return x + positions