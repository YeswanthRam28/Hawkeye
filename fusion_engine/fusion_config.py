# fusion_engine/fusion_config.py

from pydantic import BaseModel, Field
from typing import Literal


class FusionConfig(BaseModel):
    """
    Configuration for the risk fusion engine.
    All thresholds, weights, and system modes are defined here.
    """

    # Mode settings
    mode: Literal["normal", "sensitive", "aggressive", "debug"] = "normal"

    # Weighting of each subsystem in the fused score
    vision_weight: float = Field(default=0.4, ge=0, le=1)
    audio_weight: float = Field(default=0.3, ge=0, le=1)
    motion_weight: float = Field(default=0.3, ge=0, le=1)

    # Thresholds for triggering rules
    vision_threat_threshold: float = Field(default=0.6, ge=0, le=1)
    audio_anomaly_threshold: float = Field(default=0.5, ge=0, le=1)
    motion_jerk_threshold: float = Field(default=4.0)

    # Sensitivity multipliers for different modes
    sensitivity_multiplier: float = 1.0  # will be overridden by mode

    def apply_mode(self):
        """Adjust sensitivity based on selected mode."""

        if self.mode == "normal":
            self.sensitivity_multiplier = 1.0

        elif self.mode == "sensitive":
            self.sensitivity_multiplier = 0.8  # lower thresholds → more sensitive

        elif self.mode == "aggressive":
            self.sensitivity_multiplier = 0.6  # VERY sensitive system

        elif self.mode == "debug":
            self.sensitivity_multiplier = 1.0
            print("[FusionConfig] DEBUG mode active: No threshold scaling applied.")

        # Scale thresholds
        self.vision_threat_threshold *= self.sensitivity_multiplier
        self.audio_anomaly_threshold *= self.sensitivity_multiplier
        self.motion_jerk_threshold *= self.sensitivity_multiplier

    def summary(self):
        """Returns a pretty-print summary for debugging."""
        return {
            "mode": self.mode,
            "vision_weight": self.vision_weight,
            "audio_weight": self.audio_weight,
            "motion_weight": self.motion_weight,
            "vision_threat_threshold": self.vision_threat_threshold,
            "audio_anomaly_threshold": self.audio_anomaly_threshold,
            "motion_jerk_threshold": self.motion_jerk_threshold,
            "sensitivity_multiplier": self.sensitivity_multiplier,
        }
