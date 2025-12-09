import numpy as np

def compute_motion_metrics(mag):
    crowd_density = float(np.mean(mag > 2))
    panic_score = float(np.mean(mag) * 5)
    motion_variation = float(np.std(mag))

    return {
        "crowd_density": crowd_density,
        "panic_score": panic_score,
        "motion_variation": motion_variation
    }
