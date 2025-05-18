import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv, GCNConv

class GatedGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.A = nn.Linear(in_dim, out_dim)
        self.B = nn.Linear(in_dim, out_dim)
        self.C = nn.Linear(in_dim, out_dim)
        self.E = nn.Linear(in_dim, out_dim)
        self.res = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        m = self.B(x[row]) + self.C(x[col]) + self.E(edge_attr)
        m = torch.sigmoid(m) * self.A(x[row])
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, m)
        return F.relu(agg + self.res(x))

class TSPGNN(nn.Module):
    def __init__(self, hidden_dim=300, num_gcn_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(2, hidden_dim)
        self.edge_lin = nn.Linear(1, hidden_dim)

        self.transformer = TransformerConv(hidden_dim, hidden_dim, heads=1)

        self.gcn_layers = nn.ModuleList([
            GatedGCNLayer(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)
        ])

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.input_lin(x)
        edge_attr = self.edge_lin(edge_attr)

        x = self.transformer(x, edge_index)

        for layer in self.gcn_layers:
            x = layer(x, edge_index, edge_attr)

        row, col = edge_index
        edge_feat = torch.abs(x[row] - x[col])  # or try x[row] + x[col]
        edge_scores = self.edge_mlp(edge_feat).squeeze(-1)  # [E]

        return edge_scores  # logits
