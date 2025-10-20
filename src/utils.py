# src/utils.py
import os, json, numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_label_map(label_map, path="models/label_map.json"):
    with open(path, "w") as f:
        json.dump(label_map, f)

def load_label_map(path="models/label_map.json"):
    with open(path, "r") as f:
        return json.load(f)

def save_sequence(arr, out_path):
    np.save(out_path, arr, allow_pickle=False)

def load_sequence(path):
    return np.load(path)
