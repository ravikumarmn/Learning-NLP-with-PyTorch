import torch
from torch import nn
import numpy as np

class LSTM_cell(torch.nn.Module):
    """
    A simple LSTM cell network
    """
    def __init__(self, input_length=5, hidden_length=8):
        super(LSTM_cell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
# 
        # forget gate components
        self.ih_l0 = nn.Linear(self.input_length, 4*self.hidden_length)
        self.hh_l0 = nn.Linear(self.hidden_length, 4*self.hidden_length)
        self.reduce_l0 = nn.Linear(4*self.hidden_length, self.hidden_length)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh_activation = nn.Tanh()


    def input_gate(self, x, h):
        # Equation 1. input gate
        x_temp = self.ih_l0(x)#(i:4*h)
        h_temp = self.hh_l0(h)#(h:4*h)
        it = self.sigmoid(x_temp + h_temp)
        return it

    def forget(self,x,h):
        # Equation 2. forget gate
        x_temp = self.ih_l0(x)
        h_temp = self.hh_l0(h)
        ft = self.sigmoid(x_temp + h_temp)
        return ft

    def cell(self,x,h):
        # Equation 3. cell state at time t
        x_temp = self.ih_l0(x)
        h_temp = self.hh_l0(h)
        gt = self.tanh_activation(x_temp + h_temp)
        return gt

    def output(self,x,h):
        # Equation 4. output state at time t
        x_temp = self.ih_l0(x)
        h_temp = self.hh_l0(h)
        ot = self.sigmoid(x_temp + h_temp)
        return ot

    def cell_state_t(self,x,h,c_prev):
        ft = self.forget(x,h)
        it = self.input_gate(x,h)
        gt = self.cell(x,h)
        g = ft * c_prev
        s = it * gt
        ct = g + s
        return ct

    def hidden_state_t(self,x,h,c_prev):
        ot = self.output(x,h)
        ct = self.cell_state_t(x,h,c_prev)
        ct_tan = self.tanh_activation(ct)
        ht = ot * ct_tan

        return ht

    def forward(self,x,tuple_in):
        hidden_seq = list()
        (h,c) = tuple_in
        batch_sz,seq_sz,_ = x.size()

        for i in range(seq_sz):
            x_t = x[:, i, :]
            # cell state at time t
            c = self.cell_state_t(x_t,h,c)
            #hidden state at time t
            h = self.hidden_state_t(x_t,h,c) #4*h
            
            hidden_seq.append(h)

        hidden_seq = torch.cat(hidden_seq, dim=0).transpose(0, 1)
        return hidden_seq, (h, c)


if __name__ == "__main__":

    input_sz = 5
    hidden_sz = 8
    batch_sz = 1
    xt = torch.randn(1,10,5)

    custom_cell = LSTM_cell(input_sz,hidden_sz)

    k = np.sqrt(1/hidden_sz)

    init_h = torch.randn(1,batch_sz,hidden_sz)
    init_h.data.uniform_(-k,k)

    init_c_prev = torch.randn(1,batch_sz,hidden_sz)
    init_c_prev.data.uniform_(-k,k)

    tuple_in = (init_h,init_c_prev)

    hidden_seq, (h_next, c_next) = custom_cell(xt,tuple_in)
    print()


