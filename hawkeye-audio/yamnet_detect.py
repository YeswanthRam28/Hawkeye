# yamnet_detect.py

import numpy as np
import soundfile as sf
import scipy.signal
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import time
import os

# ----------------------------
# CONFIG
# ----------------------------
TARGET_SR = 16000
FRAME_STEP_S = 0.48
BLOCK_SIZE = TARGET_SR

DANGEROUS_SOUNDS = ["Gunshot", "Weapon", "Explosion", "Scream", "Siren"]
CONFIDENCE_THRESHOLD = 0.40

EVIDENCE_DIR = "evidence"
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

AUDIO_BUFFER = np.zeros(TARGET_SR * 2, dtype=np.float32)

yamnet_model = None
class_names = []
danger_indices = []


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model():
    global yamnet_model, class_names, danger_indices

    print("Loading YAMNet model...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = yamnet_model.class_map_path().numpy()
    with open(class_map_path, "r") as f:
        class_names = [line.strip() for line in f]

    danger_indices = [
        i for i, name in enumerate(class_names)
        if any(w.lower() in name.lower() for w in DANGEROUS_SOUNDS)
    ]

    print("Model loaded. Dangerous classes:", len(danger_indices))


# ----------------------------
# LOAD AUDIO FILE
# ----------------------------
def load_audio_file(filename):
    if not os.path.exists(filename):
        print("File not found:", filename)
        return None

    audio, sr = sf.read(filename)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = scipy.signal.resample_poly(audio, TARGET_SR, sr)

    return audio.astype(np.float32)


# ----------------------------
# FILE DETECTION
# ----------------------------
def run_file_detection(waveform, filename="uploaded"):
    scores, _, _ = yamnet_model(waveform)
    scores = scores.numpy()

    mean_scores = np.mean(scores, axis=0)
    top_idx = mean_scores.argmax()
    top_label = class_names[top_idx]
    confidence = mean_scores[top_idx]

    print("\n=== FILE ANALYSIS ===")
    print(f"Top sound: {top_label} ({confidence:.3f})")

    # Detect dangerous events
    events = []
    for frame in range(scores.shape[0]):
        for idx in danger_indices:
            if scores[frame, idx] >= CONFIDENCE_THRESHOLD:
                events.append((frame * FRAME_STEP_S, class_names[idx], scores[frame, idx]))

    if events:
        print("DANGEROUS SOUND DETECTED!")
        print(events[:5])
    else:
        print("No dangerous sounds detected.")


# ----------------------------
# LIVE MICROPHONE DETECTION
# ----------------------------
def process_block(indata, frames, time_info, status):
    global AUDIO_BUFFER

    audio = indata[:, 0].astype(np.float32)
    AUDIO_BUFFER[:-frames] = AUDIO_BUFFER[frames:]
    AUDIO_BUFFER[-frames:] = audio

    scores, _, _ = yamnet_model(audio)
    scores = scores.numpy()

    mean_scores = np.mean(scores, axis=0)
    top_label = class_names[mean_scores.argmax()]
    top_conf = mean_scores.max()

    print(f"Listening... {top_label} ({top_conf:.2f})")

    # Check dangerous
    for frame in range(scores.shape[0]):
        for idx in danger_indices:
            if scores[frame, idx] >= CONFIDENCE_THRESHOLD:
                print("\n⚠️  DANGEROUS SOUND DETECTED!")
                print(class_names[idx], scores[frame, idx])


def start_continuous_monitoring(stop_event=None):

    print("Microphone monitoring started.")

    def callback(indata, frames, time_info, status):
        if stop_event and stop_event.is_set():
            raise sd.CallbackStop
        process_block(indata, frames, time_info, status)

    with sd.InputStream(
        channels=1,
        samplerate=TARGET_SR,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=callback
    ):
        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                time.sleep(0.1)
        except sd.CallbackStop:
            pass

    print("Monitoring stopped.")
