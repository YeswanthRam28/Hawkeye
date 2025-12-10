# audio_anomaly.py

def detect_anomaly(audio_tensor):
    """
    Detects a simple anomaly based on the mean absolute energy of the audio chunk.
    This acts as a fast loudness threshold check.
    """
    # Calculate the Mean Absolute Value (MAV) of the audio signal
    # .abs() makes all values positive
    # .mean() gets the average amplitude
    # .item() extracts the single float value from the tensor
    energy = audio_tensor.abs().mean().item()
    
    # A simple threshold: 0.3 means 30% of max loudness.
    return energy > 0.3