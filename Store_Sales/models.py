import torch
import torch.nn as nn
from torch.utils.data import Dataset

class FFNeuralNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64, num_hidden_layers=2):
        super(FFNeuralNetwork, self).__init__()

        self.act = nn.ReLU()
        self.head = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)

        layers = []
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim ))

        self.stem = nn.Sequential(*layers)


    def forward(self, x):
        x = self.act(self.head(x))

        for layer in self.stem:
            x =  x + self.act(layer(x))

        return self.output(x)

class LSTMNeuralNetwork(nn.Module):
    def __init__(self, input_dim, endogenous_dim, endogenous_len, exogenous_dim, hidden_dim, out_dim, out_seq_len, num_layers):
        super(LSTMNeuralNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.out_seq_len = out_seq_len
        self.endogenous_dim = endogenous_dim
        self.exogenous_dim = exogenous_dim

        self.act = nn.ReLU()
        self.linear1 = nn.Linear(endogenous_dim* endogenous_len, (input_dim - exogenous_dim) * out_seq_len)
        self.linear2 = nn.Linear((input_dim - exogenous_dim) * out_seq_len, (input_dim - exogenous_dim) * out_seq_len)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        endogenous = x[0]
        exogenous = x[1]
        out = self.act(self.linear1(endogenous))
        out = self.act(self.linear2(out))

        out = out.view(-1, self.out_seq_len, self.input_dim - self.exogenous_dim)
        out = torch.cat((out, exogenous), dim=-1)

        out, _ = self.lstm(out)

        out = self.output_linear(out)

        return out
    
class LSTMDataset(Dataset):
    def __init__(self, data, endogenous_len, out_seq_len, index_col, endogenous_cols, exogenous_cols, out_cols, add_order=None):
        self.data = data.sort_values(by=[index_col, add_order] if add_order is not None else [index_col]) 
        self.add_len = data[add_order].nunique() if add_order is not None else 0
        self.add_cols = add_order if add_order is not None else None
        self.endogenous_len = endogenous_len
        self.out_seq_len = out_seq_len
        self.index_col = index_col
        self.endogenous_cols = endogenous_cols
        self.exogenous_cols = exogenous_cols
        self.out_cols = out_cols
        self.indices = {i:j for i,j in enumerate(self.data[self.index_col].unique())}

    def __len__(self):
        return len(self.indices) - self.endogenous_len - self.out_seq_len

    def __getitem__(self, idx):
        date_in = self.indices[idx]
        date_mid = self.indices[idx + self.endogenous_len -1]
        date_end = self.indices[idx + self.endogenous_len + self.out_seq_len -1]

        in_rows = (self.data[self.index_col] >= date_in) & (self.data[self.index_col] <= date_mid)
        out_rows = (self.data[self.index_col] > date_mid) & (self.data[self.index_col] <= date_end)

        endog = self.data[in_rows][self.endogenous_cols].values.reshape(-1)
        exog = self.data[out_rows][self.exogenous_cols]
        if self.add_cols is not None:
            exog = exog.iloc[[self.add_len * i for i in range(self.out_seq_len)]]
            exog = exog.values.reshape(self.out_seq_len, len(self.exogenous_cols))


        y = self.data[out_rows][self.out_cols].values
        y = y.reshape(self.out_seq_len, self.add_len)

        sample = {
            'endog': endog,
            'exog': exog,
            'label': y
        }
        return sample