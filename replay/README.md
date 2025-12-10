# replay (TimeWarp Replay Engine)

Drop-in replay engine for Hawkeye.

Files:
- engine.py: TimeWarpReplay class
- demo_trigger.py: demo script to force a trigger and save an incident

Usage:
1. import TimeWarpReplay and call .record_frame(frame, metadata) every frame.
2. When an incident is detected, call .trigger_replay(... overlay_fn=...) to save MP4, frames and JSON.
