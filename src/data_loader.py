# src/data_loader.py
import os, numpy as np, torch
from torch.utils.data import Dataset

class KeypointSequenceDataset(Dataset):
    def __init__(self, root_dir, seq_len=30):
        self.samples = []
        self.labels = []
        self.label_map = {}
        idx = 0
        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path): continue
            self.label_map[label] = idx
            for f in sorted(os.listdir(label_path)):
                if f.endswith(".npy"):
                    self.samples.append(os.path.join(label_path, f))
                    self.labels.append(idx)
            idx += 1
        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        arr = np.load(self.samples[i])
        if arr.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        else:
            arr = arr[:self.seq_len]
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(self.labels[i], dtype=torch.long)
