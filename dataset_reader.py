import torch
from torch.utils.data import Dataset

class PosDataset(Dataset):
    def __init__(self, tokens_path, tags_path, word2idx, tag2idx):
        self.sentences = []
        self.labels = []

        with open(tokens_path, 'r', encoding='utf-8') as f_tokens, \
             open(tags_path, 'r', encoding='utf-8') as f_tags:
            for token_line, tag_line in zip(f_tokens, f_tags):
                tokens = token_line.strip().split()
                tags = tag_line.strip().split()

                assert len(tokens) == len(tags), "Mismatch between tokens and tags."

                token_ids = [word2idx.get(tok, word2idx['<UNK>']) for tok in tokens]
                tag_ids = [tag2idx[tag] for tag in tags]

                self.sentences.append(token_ids)
                self.labels.append(tag_ids)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx], dtype=torch.long), \
               torch.tensor(self.labels[idx], dtype=torch.long)


def build_vocab(path):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.strip().split():
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def build_tag_vocab(path):
    tag_vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            for tag in line.strip().split():
                if tag not in tag_vocab:
                    tag_vocab[tag] = len(tag_vocab)
    return tag_vocab

def collate_fn(batch):
    sentences, labels = zip(*batch)

    max_len = max(len(seq) for seq in sentences)

    padded_sentences = torch.zeros(len(sentences), max_len, dtype=torch.long)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)

    for i, (seq, label) in enumerate(zip(sentences, labels)):
        padded_sentences[i, :len(seq)] = seq
        padded_labels[i, :len(label)] = label

    return padded_sentences, padded_labels
