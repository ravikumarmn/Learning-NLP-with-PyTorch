import torch
from torch import nn


# rnn = nn.RNNCell(10, 20)
# input = torch.randn(6, 3, 10)
# hx = torch.randn(8, 20)
# output = []
# for i in range(6):
#     hx = rnn(input[i], hx)
#     output.append(hx)
# print()
class cell_CELL(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(cell_CELL,self).__init__()
        self.input_sz = input_size
        self.hidden_sz = hidden_size
        self.ih = nn.Linear(input_size,hidden_size)
        
        self.hh = nn.Linear(input_size,hidden_size)
        self.tanh_function = nn.Tanh()



    def input_(self,x,prev):
        xt = self.ih(x)
        prev_t = self.hh(prev)
        ht = self.tanh_function(xt+prev_t)
        return ht

    def forward(self,x,prev=None):
        batch_sz,seq_sz,_ = x.size()
        hidden_seq = list()
        if prev is None:
            prev = torch.zeros(x.size(0), self.hidden_sz)
        else:
            prev = prev

        for i in range(seq_sz):
            xt = x[:,i,:]
            prev = self.input_(xt,prev)
            hidden_seq.append(prev)
        output = torch.stack(hidden_seq, dim=1)
        return output,prev



if __name__ == "__main__":
    x = torch.randn(6, 3, 10)
    prev = torch.randn(8, 20)
    # x = torch.randn(1,10,5)
    # prev = torch.randn(1,5)

    rnn = nn.RNNCell(10, 20)
    Wih = rnn.weight_ih.detach() #(20,10)
    Whh = rnn.weight_hh.detach()
    Bih = rnn.bias_ih.detach()
    Bhh = rnn.bias_hh.detach()
    cell = cell_CELL(10,20)
    # cell.ih.weight = nn.Parameter(Wih,requires_grad=True)
    # cell.hh.weight = nn.Parameter(Whh,requires_grad=True)
    # cell.ih.bias = nn.Parameter(Bih,requires_grad=True)
    # cell.hh.bias = nn.Parameter(Bhh,requires_grad=True)
    cell_dict = cell.state_dict()

    for nm,pr in rnn.named_parameters():
        wob,name = nm.split("_")
        cell_dict[f"{name}.{wob}"] = pr.clone()


        


    output,ht = cell(x,prev=None)
    # rnn = nn.RNNCell(10, 20)
    # input = torch.randn(6, 3, 10)
    outputs = []
    for i in range(3):
        hx = rnn(x[:,i,:])
        outputs.append(hx)
    outputs = torch.stack(outputs,dim=1)
    print("outputs : ",outputs[0][0])
    print("output : ",output[0][0])
    print(outputs)
        
