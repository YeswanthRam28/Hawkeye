# simple_risk_fuser.py

def fuse_risk(vision_conf, motion_norm, audio_event_strength=0.0):
    """
    Computes a simple fused risk score for each person.
    All inputs are in range 0..1.

    vision_conf: YOLO detection confidence
    motion_norm: normalized motion magnitude (0..1)
    audio_event_strength: optional audio anomaly (0..1)
    """

    # Weights (can tune later)
    w_vision = 0.6
    w_motion = 0.3
    w_audio = 0.1

    risk = (
        w_vision * float(vision_conf) +
        w_motion * float(motion_norm) +
        w_audio * float(audio_event_strength)
    )

    # ensure valid range
    return max(0.0, min(1.0, risk))
