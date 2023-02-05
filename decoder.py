import torch
from torch import nn
from modules import *
from typing import Optional, Tuple
from mask import *


class DecoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float
    ):
      super().__init__()
      self.size = size
      self.self_attn = self_attn
      self.src_attn = src_attn
      self.feed_forward = feed_forward
      self.norm1 = nn.LayerNorm(size, eps=1e-5)
      self.norm2 = nn.LayerNorm(size, eps=1e-5)
      self.norm3 = nn.LayerNorm(size, eps=1e-5)
      self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self,tgt:torch.Tensor,tgt_mask:torch.Tensor,memory:torch.Tensor,memory_mask:torch.Tensor):
        residual=tgt
        tgt=self.norm1(tgt)
    
        tgt_q = tgt
        tgt_q_mask = tgt_mask
       
       # print(tgt_mask.shape)
        x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt,tgt_q_mask)[0])
        residual=x
        x=self.norm2(x)
        x = residual + self.dropout(self.src_attn(x, memory, memory,memory_mask)[0])
        residual=x
        x=self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        
        return x

class transformerdecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 2,
        linear_units: int = 100,
        num_blocks: int = 1,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        self.norm= torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
            ) for _ in range(self.num_blocks)
        ])
    def forward(self, memory: torch.Tensor,memory_mask: torch.Tensor,ys_in_pad: torch.Tensor,ys_in_lens: torch.Tensor,r_ys_in_pad: torch.Tensor = torch.empty(0),reverse_weight: float = 0.0):
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        x,_=self.embed(tgt)
        for layer in self.decoders:
            x = layer(x, tgt_mask, memory,memory_mask)
        x=self.norm(x)
        x=self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x,torch.tensor(0.0),olens

class bitransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 2,
        linear_units: int = 100,
        num_blocks: int = 1,
        r_num_blocks: int=1,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,

    ):
       
        self.leftdecoder=transformerdecoder(vocab_size,encoder_output_size,attention_heads,linear_units,
                                           num_blocks,dropout_rate,positional_dropout_rate,self_attention_dropout_rate,src_attention_dropout_rate)
        
        self.rightdecoder=transformerdecoder(vocab_size,encoder_output_size,attention_heads,linear_units,
                                           r_num_blocks,dropout_rate,positional_dropout_rate,self_attention_dropout_rate,src_attention_dropout_rate)
    
    def forward(self, memory: torch.Tensor,memory_mask: torch.Tensor,ys_in_pad: torch.Tensor,ys_in_lens: torch.Tensor,r_ys_in_pad: torch.Tensor,
        reverse_weight: float = 0.0):
        l_x, _, olens = self.leftdecoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = torch.tensor(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.rightdecoder(memory, memory_mask, r_ys_in_pad,
                                               ys_in_lens)
        return l_x, r_x, olens

        
if __name__=='__main__':
    transformer=transformerdecoder(80,45,3,100,2)
    dummy_input=torch.ones(3,180,80)
    dummy_output=transformer(dummy_input)
    print(dummy_output.shape)
