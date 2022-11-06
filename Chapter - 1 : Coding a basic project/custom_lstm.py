import torch
from torch import nn
import numpy as np

class LSTM_cell(torch.nn.Module):
    """
    A simple LSTM cell network
    """
    def __init__(self, input_length, hidden_length):
        super(LSTM_cell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # forget gate components
        self.linear_forget_w1 = nn.Linear(self.input_length, self.hidden_length)
        self.linear_forget_r1 = nn.Linear(self.hidden_length, self.hidden_length)
        
        # input gate components
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length)

        # cell memory components
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length)

        # out gate components
        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh_activation = nn.Tanh()


    def input_gate(self, x, h):
        # Equation 1. input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        it = self.sigmoid(x_temp + h_temp)
        return it

    def forget(self,x,h):
        # Equation 2. forget gate
        x_temp = self.linear_forget_w1(x)
        h_temp = self.linear_forget_r1(h)
        ft = self.sigmoid(x_temp + h_temp)
        return ft

    def cell(self,x,h):
        # Equation 3. cell state at time t
        x_temp = self.linear_gate_w3(x)
        h_temp = self.linear_gate_r3(h)
        gt = self.tanh_activation(x_temp + h_temp)
        return gt

    def output(self,x,h):
        # Equation 4. output state at time t
        x_temp = self.linear_gate_w4(x)
        h_temp = self.linear_gate_r4(h)
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
        return ct_tan

    def forward(self,x,tuple_in):
        hidden_seq = list()
        (h,c_prev) = tuple_in
        batch_sz,seq_sz,_ = x.size()

        for i in range(seq_sz):
            x_t = x[:, i, :]
            
            # input_gate 
            it = self.input_gate(x_t,h)
            # forget gate 
            ft = self.forget(x_t,h)
            # cell
            gt = self.cell(x_t,h)
            # output gate
            ot = self.output(x_t,h)
            # cell state at time t
            c_next = self.cell_state_t(x_t,h,c_prev)
            #hidden state at time t
            h_next = self.hidden_state_t(x_t,h,c_prev)
            hidden_seq.append(h_next)

        hidden_seq = torch.cat(hidden_seq, dim=0).transpose(0, 1)
        return hidden_seq, (h_next, c_next)

