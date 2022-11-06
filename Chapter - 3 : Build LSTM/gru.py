from re import X
from turtle import forward
import torch
from torch import nn
import numpy as np

class GRU_Network(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.input_sz = input_size
        self.hidden_sz = input_size

        self.ih = nn.Linear(input_size,hidden_size)
        self.hh = nn.Linear(hidden_size,hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def reset(self,x,h_prev):
        i = self.ih(x)
        h = self.hh(h_prev)
        rt = self.sigmoid(i + h)
        return rt 

    def update(self,x,h_prev):
        i = self.ih(x)
        h = self.hh(h_prev)
        zt = self.sigmoid(i + h)
        return zt

    def new_gate(self,x,h_prev):
        i = self.ih(x)
        h = self.hh(h_prev)
        rt = self.reset(x,h_prev)
        n = i + rt
        nt = self.tanh(n*h)
        return nt

    def forward(self,x,h_prev):
        batch_sz,seq_sz,_ = x.size()

        if h_prev is None:
            h_prev = torch.zeros(batch_sz,self.hidden_sz)
        hidden_seq = list()
        for i in range(seq_sz):
            xt = x[:,i,:]
            # update
            zt = self.update(xt,h_prev)
            # new gate
            nt = self.new_gate(xt,h_prev)
            ht = 1-zt*nt + (zt*h_prev)
            hidden_seq.append(ht)
        hidden_seq = torch.stack(hidden_seq, dim=1)
        return hidden_seq,ht


if __name__ == "__main__": 
    # hidden_sz = 20
    batch_sz = 5
    seq_sz = 3
    input_sz =10
    hidden_sz = 20
    x = torch.randn(1, 3, 10)
    gru = nn.GRU(input_sz, hidden_sz)
    custom_cell = GRU_Network(input_sz,hidden_sz)
    # cell_dict = custom_cell.state_dict()

    # for nm,pr in gru.named_parameters():
    #     wob,name,_ = nm.split("_")
    #     print(wob,cell_dict[f"{name}.{wob}"].shape,pr.shape)
    #     cell_dict[f"{name}.{wob}"] = pr.clone()
    #     # print(cell_dict[f"{name}.{wob}"].shape)
    # custom_cell.load_state_dict(cell_dict)
    output, hn = custom_cell(x,h_prev = None)
    # k = np.sqrt(1/hidden_sz)
    # input = torch.randn(5, 3, 10)
    # init_h = torch.randn(1,batch_sz,hidden_sz)
    # init_h.data.uniform_(-k,k)

    # custom_cell = GRU_CELL(input_sz,hidden_sz)

    # hidden_seq,ht = custom_cell(input,init_h)
    print()

