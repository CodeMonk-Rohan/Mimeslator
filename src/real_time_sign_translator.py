# src/real_time_sign_translator.py
import cv2, numpy as np, time
import mediapipe as mp
from collections import deque
import torch
from src.model import GestureLSTM
from src.utils import load_label_map
from googletrans import Translator
import pyttsx3

SEQ_LEN = 30
MODEL_PATH = "models/gesture_lstm.pt"
LABEL_MAP_PATH = "models/label_map.json"
TARGET_LANG = "hi"   # change to desired language code
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def load_model(model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH):
    label_map = load_label_map(label_map_path)
    inv_map = {int(v): k for k,v in label_map.items()}
    num_classes = len(label_map)
    model = GestureLSTM(input_size=63, hidden_size=128, num_layers=2, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, inv_map

def kp_from_results(results):
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        kp = []
        for p in lm.landmark:
            kp.extend([p.x, p.y, p.z])
        return np.array(kp, dtype=np.float32)
    return np.zeros(63, dtype=np.float32)

def main():
    print("Using device:", DEVICE)
    model, inv_map = load_model()
    translator = Translator()
    tts = pyttsx3.init()
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    buffer = deque(maxlen=SEQ_LEN)
    last_label = None
    last_time = 0
    cooldown = 1.2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(frame_rgb)
        kp = kp_from_results(res)
        buffer.append(kp)

        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(buffer) == SEQ_LEN:
            seq = np.array(buffer)  # (seq_len,63)
            inp = torch.tensor(seq[None, :, :], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                out = model(inp)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                idx = int(probs.argmax())
                conf = float(probs[idx])
                label = inv_map[idx]
            if conf > 0.6:
                now = time.time()
                if label != last_label or (now - last_time) > cooldown:
                    # translate
                    try:
                        trans = translator.translate(label, dest=TARGET_LANG).text
                    except Exception:
                        trans = label
                    print(f"Detected: {label} ({conf:.2f}) -> {trans}")
                    tts.say(trans)
                    tts.runAndWait()
                    last_label = label
                    last_time = now
                cv2.putText(frame, f"{label} {conf:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Sign Translator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
