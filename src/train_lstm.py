# src/train_lstm.py
import torch, os, argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from src.data_loader import KeypointSequenceDataset
from src.model import GestureLSTM
from src.utils import save_label_map, ensure_dir

def train(data_dir, model_dir="models", seq_len=30, epochs=25, batch_size=8, lr=1e-3, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)
    dataset = KeypointSequenceDataset(data_dir, seq_len=seq_len)
    print("Found classes:", dataset.label_map)
    ensure_dir(model_dir)
    save_label_map(dataset.label_map, os.path.join(model_dir, "label_map.json"))
    num_classes = len(dataset.label_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GestureLSTM(input_size=63, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        print(f"Epoch {epoch}/{epochs} Loss: {total_loss/total:.4f} Acc: {correct/total:.4f}")
        torch.save(model.state_dict(), os.path.join(model_dir, f"gesture_epoch{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join(model_dir, "gesture_lstm.pt"))
    print("Saved model to", model_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="dataset_np/train")
    p.add_argument("--model_dir", default="models")
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    train(args.data_dir, args.model_dir, seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
