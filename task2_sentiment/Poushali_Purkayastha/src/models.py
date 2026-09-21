import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class MeanPoolMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids, lengths):
        x = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).to(x.dtype)
        h = self.drop(F.relu(self.fc1(pooled)))
        return self.fc2(h).squeeze(1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, kernel_sizes, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, n_filters, k) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, ids, lengths):
        x = self.emb(ids).transpose(1, 2)
        feats = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            h = F.relu(conv(x))
            pos = torch.arange(h.size(2), device=ids.device).unsqueeze(0)
            valid = pos <= (lengths.clamp(min=k) - k).unsqueeze(1)
            h = h.masked_fill(~valid.unsqueeze(1), 0.0)
            feats.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(feats, dim=1))).squeeze(1)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, ids, lengths):
        x = self.emb(ids)
        packed = pack_padded_sequence(x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.drop(h)).squeeze(1)


def build_model(mcfg, vocab_size):
    kind = mcfg["type"]
    if kind == "mean_mlp":
        return MeanPoolMLP(vocab_size, mcfg["embed_dim"], mcfg["hidden_dim"], mcfg["dropout"])
    if kind == "textcnn":
        return TextCNN(vocab_size, mcfg["embed_dim"], mcfg["n_filters"], list(mcfg["kernel_sizes"]), mcfg["dropout"])
    if kind == "bilstm":
        return BiLSTM(vocab_size, mcfg["embed_dim"], mcfg["hidden_dim"], mcfg["n_layers"], mcfg["dropout"])
    raise ValueError(f"unknown model type {kind}")
