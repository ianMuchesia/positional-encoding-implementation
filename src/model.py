import torch
import torch.nn as nn
from src.positional_encodings import SinusoidalPositionalEncoding,LearnedPositionalEncoding
from src.multihead_attention import MultiHeadAttention



class ToyModel(nn.Module):
    def __init__(self,d_model,vocab_size,max_len,pe_type):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pe_type = pe_type
        
        self.emb = nn.Embedding(vocab_size,d_model)
        
        
        self.sinPE = SinusoidalPositionalEncoding(self.d_model,self.max_len)
        self.learnedPE = LearnedPositionalEncoding(self.d_model,self.max_len)
        
        
        self.mha = MultiHeadAttention(self.d_model,4)
        
        self.classifier = nn.Linear(d_model, vocab_size)
        
        
        
    def forward(self,x):
        
        out = self.emb(x)
        
        if(self.pe_type == 'sinusoidal'):
            out = self.sinPE(out)
            
        elif(self.pe_type == 'learned'):
            out = self.learnedPE(out)
            
        else:
            out = out
            
            
        out,weights = self.mha(out,out,out)
        
        out = self.classifier(out)
        
        
        return out,weights
        
        