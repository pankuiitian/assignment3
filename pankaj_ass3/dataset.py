import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class LanguageProcessor:
    def __init__(self, data_column, vocab_path):
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            vocab = []
            
        self.idx_to_token = {0: "<pad>"}
        self.idx_to_token.update({i+1: t for i, t in enumerate(vocab)})
        self.token_to_idx = {t: i for i, t in self.idx_to_token.items()}
        
        self.SOS = self.token_to_idx.get("<s>")
        self.EOS = self.token_to_idx.get("</s>")
        self.UNK = self.token_to_idx.get("<unk>")
        
        self.data = data_column
        self.vocab_size = len(self.idx_to_token)
    
    def encode(self, text):
        return self.stoi(text) + [self.EOS]

    def stoi(self, text):
        return [self.token_to_idx.get(c, self.UNK) for c in text]
    
    def decode(self, idxs):
        filtered = []
        if isinstance(idxs, torch.Tensor):
            for idx in idxs:
                val = idx.item()
                if val != 0 and val != self.EOS:
                    filtered.append(val)
        else:
            filtered = [idx for idx in idxs if idx != 0 and idx != self.EOS]
            
        return ''.join(self.idx_to_token.get(idx, "<unk>") for idx in filtered)

    def __getitem__(self, idx):
        word = self.data.iloc[idx]
        tokens = self.stoi(word)
        return tokens + [self.EOS]

class TransliterationDataset(Dataset):
    def __init__(self, data_path, normalize=False):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                    
                if normalize:
                    source, target = parts[1], parts[0]
                    attestation = parts[2] if len(parts) > 2 else "1"
                    key = (source, attestation)
                    
                    found = False
                    for i, item in enumerate(data):
                        if item["source"] == source:
                            found = True
                            if attestation > item.get("attestation", "1"):
                                data[i] = {"source": source, "target": target, "attestation": attestation}
                            break
                            
                    if not found:
                        data.append({"source": source, "target": target, "attestation": attestation})
                else:
                    data.append({"source": parts[1], "target": parts[0]})
        
        if normalize:
            data = [{"source": d["source"], "target": d["target"]} for d in data]
                    
        self.df = pd.DataFrame(data)
        
        os.makedirs("dump", exist_ok=True)
        self.source = LanguageProcessor(self.df["source"], "dump/source_vocab.txt")
        self.target = LanguageProcessor(self.df["target"], "dump/target_vocab.txt")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = torch.tensor(self.source[idx], dtype=torch.long)
        tgt = torch.tensor(self.target[idx], dtype=torch.long)
        return src, tgt

def batch_collate(batch):
    sources, targets = [], []
    for src, tgt in batch:
        sources.append(src)
        targets.append(tgt)
        
    src_padded = pad_sequence(sources, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

def prepare_vocabularies(train_path, valid_path):
    source_chars, target_chars = set(), set()
    
    for filepath in [train_path, valid_path]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    target_chars.update(parts[0])
                    source_chars.update(parts[1])
    
    special_tokens = ["<s>", "</s>", "<unk>"]
    
    os.makedirs("dump", exist_ok=True)
    with open("dump/source_vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(special_tokens + sorted(list(source_chars))))
        
    with open("dump/target_vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(special_tokens + sorted(list(target_chars))))