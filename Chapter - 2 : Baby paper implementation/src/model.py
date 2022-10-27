import torch
import torch.nn as nn 
import pickle

class BCModel(nn.Module):
    def __init__(self,args_dict,num_dim,pretrained_embd = None):
        """
        Given embedding_matrix: numpy array with vector for all words
        return prediction ( in torch tensor format)
        """
        super(BCModel, self).__init__()

        self.embedding = nn.Embedding(num_dim,
                        embedding_dim=args_dict['EMBED_SIZE'])
        if pretrained_embd != None:
            pickle_data = pickle.load(open(pretrained_embd,'rb'))
            pickle_data = torch.tensor(pickle_data["embedding_vector"],requires_grad = True).float()
            emb_matrix = nn.Parameter(pickle_data)
            assert self.embedding.weight.shape == emb_matrix.shape
            self.embedding.weight = emb_matrix

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=args_dict['EMBED_SIZE'],out_channels=args_dict["HIDDEN_SIZE"],kernel_size=args_dict['kernel_size'],stride = args_dict["stride"])
        )
        for _ in range(6):
            self.conv_net.append(
                nn.Conv1d(in_channels=args_dict['HIDDEN_SIZE'],out_channels=args_dict["HIDDEN_SIZE"],kernel_size=args_dict['kernel_size'],stride = args_dict["stride"])
            )

        self.out = nn.Linear(args_dict["HIDDEN_SIZE"], args_dict["n_labels"])
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # pass input (tokens) through embedding layer
        x = self.embedding(x)
        hidden = self.conv_net(x.permute(0,2,1))

        out = self.out(hidden.squeeze(-1))
        out = self.sigmoid(out)
        # return output
        return out