import torch
from torch import nn
from helperFunctions import *

################################################################################################

class encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(encoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, bidirectional=False)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.max = nn.MaxPool2d(2)
        self.conv1 = torch.nn.Conv1d(in_channels=3, out_channels=12, kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        x = x.float()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.unsqueeze(x, 0).float()
        out, hidden = self.lstm(x, hidden)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim), weight.new(self.n_layers, batch_size, self.hidden_dim))
        return hidden

################################################################################################

class decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super(decoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, dropout = dropout)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

    def forward(self, decoder_hidden, input):
        batch = input.shape[0]
        input = self.embedding(input.long())
        input = input.reshape((batch, self.embedding_dim))
        input_lstm = torch.unsqueeze(input, 0)
        output, hidden = self.lstm(input_lstm, decoder_hidden)
        output = self.fc1(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim), weight.new(self.n_layers, batch_size, self.hidden_dim))
        return hidden

################################################################################################

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, word):
        batch = img.shape[0]

        # Initialize hidden states of encoder
        self.h = self.encoder.init_hidden(batch_size=batch)
        self.h = tuple([torch.nan_to_num(e, nan=0.0).data for e in self.h])

        # Iterate over pixel columns
        for i in range(0, img.shape[3]):
            row = img[:, :, :, i]
            out, self.h = self.encoder(row, self.h)
            self.h = tuple([torch.nan_to_num(e, nan=0.0).data for e in self.h])
            self.h_d = tuple([torch.nan_to_num(e, nan=0.0).data for e in self.h])

        # Ask decoder for prediction of next character for given character
        predictions=[]
        previousToken = torch.tensor([[53] for i in range(0, img.shape[0])])
        for i in range(0, word.shape[1]):
            out, self.h_d = self.decoder(self.h_d, previousToken)
            self.h_d = tuple([torch.nan_to_num(e, nan=0.0).data for e in self.h_d])
            predictions.append(out[0])
            previousToken = torch.unsqueeze(word[:, i], 1)
        predictions = torch.swapaxes(torch.stack(predictions), 0, 1)
        predictions = torch.swapaxes(predictions, 1, 2)
        return predictions