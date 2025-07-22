import torch
from torch.utils.data import DataLoader
from dataset_reader import PosDataset, build_vocab, build_tag_vocab, collate_fn
from bilstm_crf import BiLSTM_CRF
from dotenv import load_dotenv
import os

load_dotenv()
SEGMENTER = os.getenv("SEGMENTER", "baseline")
LANGUAGE = os.getenv("LANGUAGE", "nheengatu")

print(f"Language selected: {LANGUAGE} | Segmenter selected: {SEGMENTER}")

file_prefix = "_" if SEGMENTER == "baseline" else f"_{SEGMENTER}_"
tokens_train = f'data/{SEGMENTER}/{LANGUAGE}_train{file_prefix}tokens.txt'
tags_train = f'data/{SEGMENTER}/{LANGUAGE}_train{file_prefix}tags.txt'
tokens_dev = f'data/{SEGMENTER}/{LANGUAGE}_dev{file_prefix}tokens.txt'
tags_dev = f'data/{SEGMENTER}/{LANGUAGE}_dev{file_prefix}tags.txt'

# Build vocabularies
word2idx = build_vocab(tokens_train)
tag2idx = build_tag_vocab(tags_train)

print(f"Word segmenter: {SEGMENTER} | Language: {LANGUAGE}")
print(f"Vocab size: {len(word2idx)} | Tag size: {len(tag2idx)}")

# Create datasets
dataset_train = PosDataset(tokens_train, tags_train, word2idx, tag2idx)
dataset_dev = PosDataset(tokens_dev, tags_dev, word2idx, tag2idx)

# DataLoaders
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
dataloader_dev = DataLoader(dataset_dev, batch_size=32, collate_fn=collate_fn)

# Model and optimizer
model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for batch_tokens, batch_tags in dataloader_train:
        batch_tokens, batch_tags = batch_tokens.to(device), batch_tags.to(device)
        mask = batch_tokens != 0

        loss = model(batch_tokens, batch_tags, mask=mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader_train)

    # Validation
    model.eval()
    total_dev_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_tokens, batch_tags in dataloader_dev:
            batch_tokens, batch_tags = batch_tokens.to(device), batch_tags.to(device)
            mask = batch_tokens != 0

            dev_loss = model(batch_tokens, batch_tags, mask=mask)
            total_dev_loss += dev_loss.item()

            predicted_tags = model.predict(batch_tokens, mask=mask)

            # Count correct predictions (ignoring padding)
            for pred_seq, true_seq, m in zip(predicted_tags, batch_tags, mask):
                seq_length = m.sum().item()
                total_correct += (torch.tensor(pred_seq[:seq_length], device=device) == true_seq[:seq_length]).sum().item()
                total_tokens += seq_length
    avg_dev_loss = total_dev_loss / len(dataloader_dev)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0


    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | Accuracy: {accuracy:.4f}")
    # print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_loss:.4f} | Dev Loss: {avg_dev_loss:.4f}")
    # print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_loss:.4f}")

print("✅ Training complete")