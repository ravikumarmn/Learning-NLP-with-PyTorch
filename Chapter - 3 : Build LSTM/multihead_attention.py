import math
import torch 
from torch import Tensor, nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self,d_k : int):
        super(ScaledDotProductAttention,self).__init__()
        self.d_k = d_k

    def forward(self,k,q,v):
        attn = torch.matmul(q/math.sqrt(self.d_k),k.transpose(2,3)) 
        attn = F.softmax(attn,dim = -1)
        output = torch.matmul(attn,v)
        return output,attn

class MultiHeadAttention(nn.Module):
    """ Multi Head Attention """
    def __init__(self,n_heads,embedding_dim,d_k,d_v):
        """
        Args:
            n_heads :  parallel attention layers = 8
            embedding_dim : model out dimention, = 512
            d_k = dimention of keys = 64
            v_v = dimention of values = 64
        """
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        super(MultiHeadAttention,self).__init__()

        self.w_q = nn.Linear(embedding_dim,n_heads*d_k,bias = False)
        self.w_k = nn.Linear(embedding_dim,n_heads*d_k,bias = False)
        self.w_v = nn.Linear(embedding_dim,n_heads*d_k,bias = False)
        self.fc = nn.Linear(n_heads*d_v,embedding_dim,bias = False)
        self.attention = ScaledDotProductAttention(d_k = self.d_k)

    def forward(self,x):
        q,k,v = x,x,x
        b,s,e = x.size()
        d_k,d_v,n_head = self.d_k,self.d_v,self.n_heads
        sz_b,len_q,len_k,len_v = q.size(0),q.size(1),k.size(1),v.size(1)

        q = self.w_q(q).view(sz_b, len_q, n_head, d_k).transpose(1,2)
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k).transpose(1,2)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_v).transpose(1,2)

        output,attn = self.attention(q,k,v)
        output = output.view(b,s,-1)
        output = self.fc(output)
        return output,attn
          
class PositionalEncoding(nn.Module):
    def __init__(self,embedding_dim,max_seq_len):
        super().__init__()

        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term_even = torch.pow(torch.tensor(10000),(torch.arange(0, embedding_dim, 2).float())/embedding_dim)
        div_term_odd = torch.pow(torch.tensor(10000),(torch.arange(1, embedding_dim, 2).float())/embedding_dim)
        pe[:, 0::2] = torch.sin(position / div_term_even)
        pe[:, 1::2] = torch.cos(position / div_term_odd)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe",pe)

    def forward(self,x):
         x = x + self.pe[:, : x.size(1)]
         return x

class PositionFeedForward(nn.Module):
    def __init__(self,embedding_dim,hidden_sz):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim,hidden_sz)
        self.linear2 = nn.Linear(hidden_sz,embedding_dim)

    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,params):
        super(EncoderLayer,self).__init__()

        self.position_encoding = PositionalEncoding(params["embedding_sz"],params["seq_len"])
        self.head_attention = MultiHeadAttention(params["n_heads"],params["embedding_sz"],params["d_k"],params["d_v"])
        self.layer_norm = nn.LayerNorm(params["embedding_sz"])
        self.layer_norm1 = nn.LayerNorm(params["embedding_sz"])
        self.postionfeedforward = PositionFeedForward(params["embedding_sz"],params["hidden_sz"])
        self.postionfeedforward1 = PositionFeedForward(params["embedding_sz"],params["hidden_sz"])


    def forward(self,input_seq):
        position_encoded = self.position_encoding(input_seq)
        head_attention = self.head_attention(position_encoded)

        added = torch.add(position_encoded,head_attention[0])

        layer_norm = self.layer_norm(added)
        position_ff = self.postionfeedforward(layer_norm)

        added1 = torch.add(position_ff,layer_norm)

        layer_norm1 = self.layer_norm1(added1)
        
        return layer_norm1

class Config:
    batch_sz = 1
    embedding_sz = 3
    seq_len = 4
    hidden_sz = 2048
    n_heads = 8
    d_k = 64
    d_v = 64

if __name__ == "__main__":

    my_input = torch.tensor([[[ 1.4359, -0.3748, -1.0442],
                              [-0.1581,  0.2255,  2.0028],
                              [-0.0647,  0.0616,  0.0286],
                              [ 1.0268, -0.3816,  1.7170]]])

    shifted_input = torch.tensor([[[-0.1581,  0.2255,  2.0028],
                                   [-0.0647,  0.0616,  0.0286],
                                   [ 1.0268, -0.3816,  1.7170],
                                   [-0.5707, -0.0059,  2.7354]]])
    input_seq = my_input
    params = {i:j for i,j in Config.__dict__.items() if "__" not in i}
    # input_seq = torch.randn(params['batch_sz'],params['seq_len'],params['embedding_sz'])
    encoder_layer = EncoderLayer(params)
    encoderlayer_output = encoder_layer(input_seq)

    print()