
import torch
import torch.nn as nn
from collections import OrderedDict

class CrossEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1):
        super().__init__()
        self.mha_lr=MHA(embed_dim,num_heads,attn_dropout,res_dropout)
        self.mha_rl=MHA(embed_dim,num_heads,attn_dropout,res_dropout)
        self.ffn=FFN(embed_dim,relu_dropout,res_dropout)

    def forward(self, h_left, h_right): # sl bs n

        h_right_l, attn_weight_matrix = self.mha_rl(h_right,h_left,h_left, reverse=True)
        h_right_l = self.ffn(h_right_l)

        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        h_left_r, attn_weight_matrix = self.mha_lr(h_left,h_right,h_right, add_attn_weights=attn_weight_matrix)
        h_left_r = self.ffn(h_left_r)

        return h_left_r, h_right_l


class SelfEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1):
        super().__init__()
        self.embed_dim=embed_dim
        self.mha_l=MHA(embed_dim,num_heads,attn_dropout,res_dropout)
        self.mha_r=MHA(embed_dim,num_heads,attn_dropout,res_dropout)
         
        self.ffn=FFN(embed_dim,relu_dropout,res_dropout)

    def forward(self, h_l, h_r=None, draw=False): # sl bs n
        h_l=self.ffn(self.mha_l(h_l)[0])
        if h_r != None:
            h_r=self.ffn(self.mha_r(h_r)[0])
            return h_l, h_r
        return h_l
        


class MHA(nn.Module):
    def __init__(self,embed_dim,num_heads,attn_dropout,res_dropout,input_dim=None,generalized_attention=False,dim_head_down=1):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.res_dropout=res_dropout
        self.generalized_attention=generalized_attention
        self.dim_head=int(embed_dim/num_heads/dim_head_down)

        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, input_dim=input_dim)

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        input_dim=self.embed_dim if input_dim is None else input_dim
        self.layer_norm_kv = nn.LayerNorm(input_dim)

    def forward(self,x,x_k=None,x_v=None, reverse=False, add_attn_weights=None):
        sl,bs,_=x.shape # sl bs n
        residual = x
        x = self.layer_norm(x)
        c=x if x_k is None else x_k

        if x_k is not None: c = self.layer_norm_kv(c)

        x, attn_weights = self.attn(x, c, c, add_attn_weights=add_attn_weights, reverse=reverse)

        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = x.squeeze(1).reshape(sl,bs,-1)
        x = residual + x
        
        return x, attn_weights

class FFN(nn.Module):
    def __init__(self, embed_dim,relu_dropout,res_dropout, is_gated=False):
        super().__init__()
        self.embed_dim=embed_dim
        self.relu_dropout=relu_dropout
        self.res_dropout=res_dropout
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight V 
            self.linear_v = nn.Linear(embed_dim, 4 * embed_dim)

    def forward(self,x):
        residual = x
        x = self.layer_norm(x)
        # f(xW1+b1)
        g = F.relu(self.fc1(x))
        # If gated, f(xW1+b1)⊗(xV+b)
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        # Apply dropout
        x = F.dropout(x, p=self.relu_dropout, training=self.training)

        # (f(xW1+b1)⊗(xV+b))W2+b2 or f(xW1+b1)W2+b2
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        return x

    
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias: nn.init.constant_(m.bias, 0.)
    return m


