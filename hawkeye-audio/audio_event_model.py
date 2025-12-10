# audio_event_model.py

import torch
from torch import nn

def load_audio_model():
    """
    Loads a dummy PyTorch model and a class map for testing the pipeline.
    In a real project, this would load a trained model (e.g., a CNN).
    """
    
    # Dummy classifier class
    class DummyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # The input (64) matches the n_mels=64 used in the pipeline
            self.fc = nn.Linear(64, 10) 

        def forward(self, x):
            # In a real model, you would process the Mel spectrogram (x)
            # For now, it returns random scores to simulate a prediction.
            # We assume 10 output classes.
            return torch.rand(10) 

    model = DummyClassifier()
    # Dummy class names. You'll replace these with "Gunshot", "Scream", etc.
    class_map = {
        0: "speech", 
        1: "music", 
        2: "silence", 
        3: "clapping", 
        4: "whistle", 
        5: "noise",
        6: "unknown_1",
        7: "unknown_2",
        8: "unknown_3",
        9: "unknown_4"
    }

    return model, class_map