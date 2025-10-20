# src/extract_keypoints.py
import cv2, mediapipe as mp, numpy as np, os
from pathlib import Path
from src.utils import ensure_dir, save_sequence

mp_hands = mp.solutions.hands

def extract_from_video(video_path, seq_len=30):
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(video_path))
    seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(frame_rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            keypoints = []
            for p in lm.landmark:
                keypoints.extend([p.x, p.y, p.z])
            seq.append(np.array(keypoints, dtype=np.float32))
        else:
            seq.append(np.zeros(63, dtype=np.float32))
    cap.release()
    hands.close()
    # normalize length: pad or trim
    arr = np.array(seq)
    if arr.shape[0] == 0:
        arr = np.zeros((seq_len, 63), dtype=np.float32)
    elif arr.shape[0] < seq_len:
        pad = np.zeros((seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, pad])
    else:
        arr = arr[:seq_len]
    return arr

def extract_dataset(raw_dir="dataset/raw", out_np_dir="dataset_np/train", seq_len=30):
    raw_dir = Path(raw_dir)
    out_np_dir = Path(out_np_dir)
    ensure_dir(out_np_dir)
    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir(): continue
        target_dir = out_np_dir / label_dir.name
        ensure_dir(target_dir)
        for video in sorted(label_dir.glob("*.mp4")):
            out_file = target_dir / (video.stem + ".npy")
            if out_file.exists():
                print("Skipping existing:", out_file)
                continue
            arr = extract_from_video(video, seq_len=seq_len)
            save_sequence(arr, str(out_file))
            print("Saved", out_file, "shape", arr.shape)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="dataset/raw")
    p.add_argument("--out_np", default="dataset_np/train")
    p.add_argument("--seq_len", type=int, default=30)
    args = p.parse_args()
    extract_dataset(args.raw_dir, args.out_np, seq_len=args.seq_len)
