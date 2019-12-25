import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.linear_key = nn.Linear(key_size, hidden_size, bias=False)
        self.linear_query = nn.Linear(query_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size=hidden_size

    def score(self, query, key, softmax_dim=-1, mask=None):
        attn=self.matching(query, key, mask)

        attn = F.softmax(attn, dim=softmax_dim)

        return attn


    def matching(self, query, key, mask=None):
        wq = self.linear_query(query)
        wq = wq.unsqueeze(-2)

        uh = self.linear_key(key)
        uh = uh.unsqueeze(-3)

        wuc = wq + uh

        wquh = torch.tanh(wuc)

        attn = self.v(wquh).squeeze(-1)

        if mask is not None:
            attn = attn.masked_fill(1-mask, -float('inf'))

        return attn

    def forward(self, query, key, value, mask=None):

        attn = self.score(query, key, mask=mask)
        h = torch.bmm(attn.view(-1, attn.size(-2), attn.size(-1)), value.view(-1, value.size(-2), value.size(-1)))

        return h.view(list(attn.size())[:-2]+[attn.size(-2), -1]), attn