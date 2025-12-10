# explainability/audio_graph.py
import matplotlib.pyplot as plt
import numpy as np

def generate_audio_peak_graph(audio_signal, sample_rate, save_path, normalize_to=1500):
    """
    Generates a frequency spectrum (FFT) graph.
    - audio_signal : 1D numpy array
    - sample_rate  : int (e.g., 16000)
    - save_path    : file path to save PNG
    - normalize_to : max magnitude after normalization (keeps ticks 0,500,1000,1500)
    """
    n = len(audio_signal)
    if n == 0:
        raise ValueError("Empty audio signal")

    freq = np.fft.rfftfreq(n, 1 / sample_rate)
    magnitude = np.abs(np.fft.rfft(audio_signal))

    # Normalize magnitude to 0..normalize_to
    max_mag = magnitude.max() if magnitude.max() > 0 else 1.0
    magnitude = (magnitude / max_mag) * normalize_to

    plt.figure(figsize=(6, 2.5))
    plt.plot(freq, magnitude, linewidth=1)
    plt.title("Audio Frequency Spectrum", fontsize=11)
    plt.xlabel("Frequency (Hz)", fontsize=9)
    plt.ylabel("Magnitude", fontsize=9)
    plt.xlim(0, sample_rate // 2)
    plt.yticks([0, 500, 1000, 1500])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
