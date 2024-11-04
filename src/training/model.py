# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

import torch
import torch.nn as nn
import torchsnooper

DEVICE = torch.device('cuda:1')  # None


class BiLinear(nn.Module):
    def __init__(self, input_size, hidden_size=None, output_size=5):
        super(BiLinear, self).__init__()
        hidden_size = input_size if hidden_size is None else hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.actv = nn.PReLU()
    def forward(self, x):
        out = self.linear2(self.actv(self.linear1(x)))  # (G, 5)
        return out.T


'''
class LinearTest(nn.Module):
    def init(self, input_size):
        super(LinearTest, self).__init__()
        self.ln = BiLinear(input_size=2*input_size, hidden_size=input_size)
    def forward(self, x, y):
        inpt = torch.cat([x, y], dim=0).T
        out = self.ln(inpt)
        return out


class LSTM_pure_model(nn.Module):
    # seq_length : C, batch_size : G, embedding_size : 2
    def __init__(self, input_size=2, hidden_size=5):
        super(LSTM_pure_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1)
        # self.apply(rand_init) ???
        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param,-0.1,0.1)

    def forward(self, x):  # x (C, G, 2)
        x, h_c = self.lstm(x)
        return x[-1].T  # x (C, G, 5) -> (G, 5) -> (5, G)

    # def initialize(self):
    #    h_c = [torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)]  # c_n.shape==(num_layers*num_directions,batch_size,hidden_size)
    #    return h_c
'''

# @torchsnooper.snoop()
class LSTM_model(nn.Module):
    def __init__(self, input_size, lstm_hid_size, linear_hid_size, output_size, num_layers, bi_di):
        super(LSTM_model, self).__init__()
        # self.input_size = input_size
        # self.hid_size = hidden_size
        # self.output_size = output_size
        lstm_out_size = lstm_hid_size // 2 if bi_di else lstm_hid_size
        self.bi = bi_di
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_out_size, num_layers=num_layers, bidirectional=self.bi)  # 1 or 2 ?
        self.linear = BiLinear(input_size=lstm_hid_size, hidden_size=linear_hid_size, output_size=output_size)

        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param, -1, 1)

    def forward(self, x):  # x (C, G, 2)
        x, hc = self.lstm(x)
        if self.bi:
            x = (x[0] + x[-1]) / 2
        else:
            x = x[-1]  # (G, 10)
        x = self.linear(x)  # (G, 5)
        return x



class LSTM_Attn(nn.Module):
    def __init__(self, input_size, lstm_hid_size, linear_hid_size, output_size, num_layers, bi_di, attn_size=4):
        super(LSTM_Attn, self).__init__()
        lstm_out_size = lstm_hid_size // 2 if bi_di else lstm_hid_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_out_size, num_layers=num_layers, bidirectional=bi_di)  # 1 or 2 ?
        self.actv = nn.PReLU()
        self.linear = BiLinear(input_size=lstm_hid_size, hidden_size=linear_hid_size, output_size=output_size)

        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=0)
        self.w_attn = torch.randn(lstm_hid_size, 1, attn_size, requires_grad=True).to(DEVICE)

        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param, -1, 1)
    
    # @torchsnooper.snoop()
    def forward(self, x):  # x (C, G, 2)
        x, hc = self.lstm(x)
        # x = x[-1]  # (G, 10)
        hx = self.tanh(x)
        weights = torch.einsum("pqr,rqs->pqs", [hx, self.w_attn.repeat(1, x.shape[1], 1)])
        weights = self.soft(weights)
        x = torch.einsum("pqr,pqs->sqr", [x, weights]).sum(0)  # (G, lstm_hid_size)

        x = self.actv(x)
        x = self.linear(x)  # (5, G)
        return x




class LSTM_VAE(nn.Module):
    def __init__(self, input_size=2, lstm_hid_size=10, linear_hid_size=16, output_size=10, num_layers=2, mu0=0):
        super(LSTM_VAE, self).__init__()
        self.mu0 = mu0
        # self.input_size = input_size
        # self.hidden_size = hidden_size  # linear_hid = 8 or 16 ?
        # self.output_size = output_size  # (mu * 5, sigma * 5)
        self.encoder = LSTM_model(input_size=input_size, lstm_hid_size=lstm_hid_size, linear_hid_size=linear_hid_size, output_size=output_size, num_layers=num_layers)
        self.relu = nn.ReLU()

    def reparameterize(self, x):
        mu, std = x[:5], x[5:]
        eps = torch.rand_like(std)
        return eps * std + mu
        # return self.relu(eps * std + mu)

    # @torchsnooper.snoop()
    def contribution_loss(self, x):
        loss = (x[:5] - self.mu0) ** 2 + x[5:] ** 2 - torch.log(x[5:] ** 2) - 1  # (G, 5)
        return 0.5 * loss[:3].sum().item()

    def forward(self, x, mode='train'):
        x = self.encoder(x)  # (C, G, 2) -> (10, G)

        # x[:5] = torch.abs(x[:5])
        
        if mode == 'train':
            return self.reparameterize(x), self.contribution_loss(x)
        elif mode == 'avg':
            return x  # x[:5], x[5:]
        else:
            return self.reparameterize(x)



class Log_LSTM_VAE(nn.Module):
    def __init__(self, input_size=2, lstm_hid_size=10, linear_hid_size=16, output_size=10, num_layers=2, mu0=0, bi_di=False):
        super(Log_LSTM_VAE, self).__init__()
        self.mu0 = mu0
        self.cls = int(output_size / 2)
        # self.input_size = input_size
        # self.hidden_size = hidden_size  # linear_hid = 8 or 16 ?
        # self.output_size = output_size  # (mu * 5, sigma * 5)
        self.encoder = LSTM_Attn(input_size=input_size, lstm_hid_size=lstm_hid_size, linear_hid_size=linear_hid_size, output_size=output_size, num_layers=num_layers, bi_di=bi_di)

    # @torchsnooper.snoop()
    def reparameterize(self, x):
        mu, std = x[:self.cls], x[self.cls:]
        eps = torch.rand_like(std)
        # return exp * std + mu
        return eps * torch.exp(0.5 * std) + torch.exp(mu)  # put square to mu or reparam

    def contribution_loss(self, x):
        # loss = (x[:5] - self.mu0) ** 2 + x[5:] ** 2 - torch.log(x[5:] ** 2) - 1
        loss = (x[:self.cls] - self.mu0) ** 2 + torch.exp(x[self.cls:]) - x[self.cls:] - 1  # (G, 5)
        return loss.sum().item() / 2

    def forward(self, x, mode='train', mu0=None):
        if mu0 is not None:
            self.mu0 = mu0
        x = self.encoder(x)  # (C, G, 2) -> (6, G)

        if mode == 'train':
            return self.reparameterize(x), self.contribution_loss(x)  # torch.exp
        elif mode == 'avg':
            return torch.exp(x[:self.cls]) 
        else:
            return self.reparameterize(x)



'''
class LSTM_expVAE(nn.Module):
    def __init__(self, input_size=2, lstm_hid_size=4, linear_hid_size=4, output_size=3, num_layers=1, lamd0=1):
        super(LSTM_expVAE, self).__init__()
        self.lamd0 = lamd0
        self.encoder = LSTM_model(input_size=input_size, lstm_hid_size=lstm_hid_size, linear_hid_size=linear_hid_size, output_size=output_size, num_layers=num_layers)

    # @torchsnooper.snoop()
    def reparameterize(self, x, aux=1):  # aux for magnifying 
        lamd = x ** 2  # torch.exp(x)
        eps = torch.rand_like(x)
        return torch.log(1 - eps ** aux) / lamd  # lamda * torch.exp(-lamd * eps)  # (3, G)

    def contribution_loss(self, x):
        lamd = x ** 2
        loss = self.lamd0 / lamd - torch.log(self.lamd0 / lamd) - 1  # (3, G)
        return loss.sum().item() / 2

    def forward(self, x, mode='train'):
        x = self.encoder(x)  # (C, G, 2) -> (3, G)
        if mode == 'train':
            return self.reparameterize(x), self.contribution_loss(x)
        elif mode == 'param':
            return x
        else:
            return self.reparameterize(x)




model = Log_LSTM_VAE(bi_di=True)
x = torch.zeros(33, 106, 2)
y, z = model(x)
print(y.shape, z)
'''
