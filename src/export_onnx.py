# src/export_onnx.py
import torch
from src.model import GestureLSTM

def export(model_path="models/gesture_lstm.pt", onnx_path="models/gesture_lstm.onnx", seq_len=30):
    # load label map to count classes
    import json
    with open("models/label_map.json", "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    model = GestureLSTM(input_size=63, hidden_size=128, num_layers=2, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, seq_len, 63)
    torch.onnx.export(model, dummy, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])
    print("Exported ONNX to", onnx_path)

if __name__ == "__main__":
    export()
