import sounddevice as sd
import numpy as np

duration = 1
sample_rate = 16000

print("Testing microphone... speak now.")

audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate,
               channels=1, dtype='float32')
sd.wait()

if np.abs(audio).mean() < 0.001:
    print("❌ Mic seems silent or wrong input device.")
else:
    print("✅ Mic is working!")
