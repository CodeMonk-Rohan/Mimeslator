# src/capture_dataset.py
import cv2, os, time
from pathlib import Path
from src.utils import ensure_dir

def capture_for_label(label, out_dir="dataset/raw", samples=30, duration=3, fps=20):
    """
    Capture `samples` small videos for a single label.
      - duration: seconds per sample (3 recommended)
      - fps: frames per second recorded
    """
    out_dir = Path(out_dir) / label
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(f"Recording {samples} samples for label: {label}. Press 'q' to abort early.")
    for i in range(samples):
        filename = out_dir / f"{label}_{i+1:03d}.mp4"
        out = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))
        start = time.time()
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            cv2.imshow(f"Recording {label} ({i+1}/{samples})", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                return
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Done capturing for", label)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--duration", type=float, default=2.5)
    args = parser.parse_args()
    capture_for_label(args.label, samples=args.samples, duration=args.duration)
