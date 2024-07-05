from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BertMlp(nn.Module):
    def __init__(self, bert, mlp):
        super().__init__()
        self.bert = bert
        self.mlp = mlp

    def forward(self, **x):
        cls_embed = self.bert(**x)["pooler_output"]
        return self.mlp(cls_embed)
