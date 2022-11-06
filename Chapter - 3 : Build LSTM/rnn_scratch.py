import torch
from torch import nn

class MyRNN(nn.Module):
    '''Building RNN using PyTroch'''
    def __init__(self, input_size, hidden_size):
        '''
        Args : 
            input_size :Size of embeddings. (batch,seq,embeddings)
            hidden_size : Hidden size, to transform the embedding to this shape
        
        Returns :
            output : output features (h_t) from the last layer of the RNN
            hn  : final hidden state
        '''        
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
    def calulate_ht(self,x,prev):
        wih = self.ih(x)
        whh = self.hh(prev)
        combined = torch.add(wih, whh)
        hidden_state = torch.tanh(combined)
        return hidden_state

    def forward(self, x):
        batch_sz,seq_sz,_ = x.size()
        prev = torch.zeros(batch_sz, self.hidden_size)
        hidden_seq = list()
        for i in range(seq_sz):
            xt = x[:,i,:]
            prev = self.calulate_ht(xt,prev)
            hidden_seq.append(prev)
        hn = hidden_seq[-1].view(1,batch_sz,-1)
        output = torch.stack(hidden_seq,dim = 1).view(batch_sz,seq_sz,-1)
        return output,hn

class Config:
    hidden_size = 256
    input_sz = 10
    output_sz = 5
    batch_sz = 1
    seq_len = 8

if __name__ == "__main__":
    params = {i:j for i,j in Config.__dict__.items() if "__" not in i}
    rnn = nn.RNN(params["input_sz"], params["hidden_size"])
    model = MyRNN(params["input_sz"], params["hidden_size"])
    model_dict = model.state_dict()

    for nm,pr in rnn.named_parameters():
        wob,name,_ = nm.split("_")
        model_dict[f"{name}.{wob}"] = pr.clone()

    # model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    inputs = torch.randn(params["batch_sz"],params["seq_len"],params["input_sz"])
    my_rnn_output,my_rnn_hn = model(inputs)
    pytorch_rnn_output,pytorch_rnn_hn = model(inputs)
    print("My RNN model hidden outputs from the last layer of 10 samples : {}".format(my_rnn_output[0,0,:10].detach().numpy()))
    print("PyTorch RNN model hidden outputs from the last layer of 10 samples : {}".format(pytorch_rnn_output[0,0,:10].detach().numpy()))
    print("-"*50)
    print("My RNN model final hidden state for each element in the batch : {}".format(my_rnn_hn[0,0,:10].detach().numpy()))
    print("PyTorch RNN model final hidden state for each element in the batch : {}".format(pytorch_rnn_hn[0,0,:10].detach().numpy()))
