"""
Author: Sandeep Kumar Suresh
        EE23S059

The code contains the model class

"""
import torch
import torch.nn as nn
import torch.optim as optim

# class Classifier(nn.Module):
#     def __init__(self, seq_model, input_size, hidden_size, num_layers, num_classes,dropout):
#         super(Classifier, self).__init__()

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.seq_model = seq_model

#         if self.seq_model == 'RNN':
#             self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         else:    
#             self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         self.dropout = nn.Dropout(p=dropout)
#         self.fc = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):

#         x = x.float()

#         if self.seq_model == 'RNN':
#             h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
#             out,_ = self.rnn(x,h0)
#         else:
#             h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#             c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#             out, _ = self.lstm(x, (h0, c0))

#         out = self.dropout(out) 
#         out = self.fc(out[:, -1, :])
        
#         return out


class Classifier(nn.Module):
    def __init__(self, seq_model, input_size, hidden_size, num_layers, num_classes, dropout):
        super(Classifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model = seq_model

        if self.seq_model == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.seq_model == 'LSTM':
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.seq_model == 'BiLSTM':
            self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Invalid seq_model type. Choose from 'RNN', 'LSTM', or 'BiLSTM'.")

        # Adjust the input size for the fully connected layer if using BiLSTM
        self.fc_input_size = hidden_size * 2 if self.seq_model == 'BiLSTM' else hidden_size

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.fc_input_size, num_classes)
    
    def forward(self, x):
        x = x.float()

        if self.seq_model == 'RNN':
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, h0)
        elif self.seq_model == 'LSTM':
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
        elif self.seq_model == 'BiLSTM':
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.bilstm(x, (h0, c0))

        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # Take the last time-step for classification
        
        return out


class EarlyStopping:

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0