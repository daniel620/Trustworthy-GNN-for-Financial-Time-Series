import torch
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.models import GIN

from torch.nn import Conv1d, LSTM

def calculate_output_length(input_length, kernel_size, stride, padding=0, dilation=1):
    output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    return output_length

# class TempEmbedder(torch.nn.Module):

class TempGNN(torch.nn.Module):
    def __init__(self, input_length, temp_out_dim=5, temp_kernel_size=7, temp_stride=3, temp_emb_dim=32, mpnn=GIN):
        super().__init__()
        self.window_size = input_length
        self.temp_conv = Conv1d(in_channels=1,
                                    out_channels=temp_out_dim,
                                    kernel_size=temp_kernel_size,
                                    stride=temp_stride)
        output_length = calculate_output_length(input_length, temp_kernel_size, temp_stride)
        self.fc = torch.nn.Linear(output_length * temp_out_dim, temp_emb_dim)

        # self.lstm = LSTM(input_size=1, hidden_size=temp_emb_dim, num_layers=2, batch_first=True)
        # self.fc = torch.nn.Linear(temp_emb_dim, temp_emb_dim)

        self.mpnn = mpnn(temp_emb_dim, temp_emb_dim//2, num_layers=2)
        # self.conv1 = graph_conv(temp_emb_dim, temp_emb_dim//2)
        # self.conv2 = graph_conv(temp_emb_dim//2, temp_emb_dim//4)
        # self.conv1 = GCNConv(temp_emb_dim, temp_emb_dim//2)
        # self.conv2 = GCNConv(temp_emb_dim//2, temp_emb_dim//4)
        self.readout = torch.nn.Linear(temp_emb_dim//2, 1)
    def forward(self, x, edge_index):
        # x: [batch_size, num_nodes, time_length]
        if len(x.size()) == 3:
            batch_size, num_nodes, time_length = x.size()
            x = x.view(-1, 1, time_length) # 1dconv
            x = self.temp_conv(x)
            x = F.relu(x)
            # x = x.view(-1, time_length, 1) # lstm
            # _, (hn, _) = self.lstm(x)
            # x = hn[-1].squeeze(0)
            x = x.view(batch_size, num_nodes, -1)
            x = self.fc(x)
            x = self.mpnn(x, edge_index)
            # x = self.conv1(x, edge_index)
            # x = F.relu(x)
            # x = self.conv2(x, edge_index)
            # x = F.relu(x)
            x = self.readout(x)
        elif len(x.size()) == 2:
            # x: [num_nodes, time_length]
            num_nodes, time_length = x.size()
            x = x.view(num_nodes, 1, -1)
            x = self.temp_conv(x)
            x = F.relu(x)
            x = x.view(num_nodes, -1)
            # x = self.fc(x)
            # x = F.relu(x)
            # x = self.conv1(x, edge_index)
            # x = F.relu(x)
            # x = self.conv2(x, edge_index)
            # x = F.relu(x)

            x = self.fc(x)
            x = self.mpnn(x, edge_index)
            x = self.readout(x)
        
        return x
    
if __name__ == "__main__":
    # Test the TempGNN model
    model = TempGNN(input_length=100)
    x = torch.randn(32, 10, 100)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    out = model(x, edge_index)
    print(out.size())