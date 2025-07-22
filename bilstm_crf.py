import torch
import torch.nn as nn
from TorchCRF import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, pad_idx=0):
        super(BiLSTM_CRF, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, tags=None, mask=None):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            best_path = self.crf.decode(emissions, mask=mask)
            return best_path

    def predict(self, sentences, mask=None):
        with torch.no_grad():
            embeddings = self.embedding(sentences)
            lstm_out, _ = self.lstm(embeddings)
            emissions = self.hidden2tag(lstm_out)
            best_path = self.crf.decode(emissions, mask=mask)
            return best_path

    def compute_loss(self, sentences, tags, mask=None):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss
