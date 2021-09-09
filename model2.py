import torch
from torch import nn

################################################################################################

class encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(encoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, dropout = dropout, bidirectional=True)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2400, 1200)
        self.fc2 = nn.Linear(1200, embedding_dim)
        self.max = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        x = x.float()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.unsqueeze(x, 0)
        out, hidden = self.lstm(x, hidden)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim), weight.new(self.n_layers*2, batch_size, self.hidden_dim))
        return hidden

################################################################################################

class decoder_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, layersEnc):
        super(decoder_Attention, self).__init__()
        self.lstm = nn.LSTM(embedding_dim+hidden_dim, hidden_dim, num_layers = n_layers, dropout = dropout)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attn = nn.Linear(layersEnc*2, 1)
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

    def forward(self, decoder_hidden, encoder_outputs, input):
        batch = input.shape[0]
        input = self.embedding(input.long())
        input = input.reshape((batch, self.embedding_dim))
        encoder_outputs = torch.swapaxes(encoder_outputs, 0, 1)
        encoder_outputs = torch.swapaxes(encoder_outputs, 1, 2)
        weights = self.attn(encoder_outputs).reshape((encoder_outputs.shape[0], encoder_outputs.shape[1]))
        input_lstm = torch.cat((weights, input), dim=1)
        input_lstm = torch.unsqueeze(input_lstm, 0)
        output, hidden = self.lstm(input_lstm, decoder_hidden)
        output = self.dropout(output)
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

        #Initialize hidden states
        self.h = self.encoder.init_hidden(batch_size=batch)
        self.h = tuple([torch.nan_to_num(e, nan=0.0).data for e in self.h])
        self.h_d = self.decoder.init_hidden(batch_size=batch)
        self.h_d = tuple([torch.nan_to_num(e, nan=0.0).data for e in self.h_d])

        # Feed batch into encoder
        out, self.h = self.encoder(img, self.h)
        self.h = tuple([torch.nan_to_num(e, nan=0.0).data for e in self.h])

        predictions=[]
        hiddenEncoder = self.h[0]

        # For each character ask decoder for prediction of nect character
        previousToken = torch.tensor([[53] for i in range(0, img.shape[0])])
        for i in range(0, word.shape[1]):
            out, self.h_d = self.decoder(self.h_d, hiddenEncoder, previousToken)
            self.h_d = tuple([torch.nan_to_num(e, nan=0.0).data for e in self.h_d])
            predictions.append(out[0])
            previousToken = torch.unsqueeze(word[:, i], 1)
        predictions = torch.swapaxes(torch.stack(predictions), 0, 1)
        predictions = torch.swapaxes(predictions, 1, 2)
        return predictions