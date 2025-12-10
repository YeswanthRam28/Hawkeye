import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample

# Load YAMNet model from TF-Hub
print("🔄 Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_names = yamnet_model.class_names.numpy()

SAMPLE_RATE = 16000

def preprocess_audio(audio, orig_sr):
    if orig_sr != SAMPLE_RATE:
        target_len = int(SAMPLE_RATE * (len(audio) / orig_sr))
        audio = resample(audio, target_len).astype(np.float32)
    return np.array(audio, dtype=np.float32)

def classify_audio(audio, orig_sr):
    # Preprocess audio
    audio = preprocess_audio(audio, orig_sr)

    # Run model
    scores, embeddings, spectrogram = yamnet_model(audio)

    # Average scores over frames
    mean_scores = tf.reduce_mean(scores, axis=0)

    # Pick best class
    class_idx = int(tf.argmax(mean_scores))
    class_name = class_names[class_idx].decode("utf-8")
    confidence = float(mean_scores[class_idx])

    return class_name, confidence
