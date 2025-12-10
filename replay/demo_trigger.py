# replay/demo_trigger.py

import cv2
import time
from .engine import ReplayEngine

"""
Replay Demo v2.0
----------------
✔ Auto-trigger after buffer fills
✔ Manual trigger by pressing 'T'
✔ Saves clip + frames + evidence.json
"""

# CONFIG
RISK_TRIGGER = 0.50
MIN_BUFFER_TIME = 3.0
COOLDOWN = 6.0


def demo_run():
    print("=== TimeWarp Replay Demo v2.0 ===")
    print("Press 'T' for manual trigger.")
    print("Press 'ESC' to exit.")
    print("-----------------------------------")

    cap = cv2.VideoCapture(0)
    engine = ReplayEngine(pre_seconds=5, post_seconds=5, fps=20)

    last_trigger = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fake placeholder risk (replace with real pipeline output)
        fake_meta = {"risk_overall": 0.4}

        info = engine.add_frame(frame, fake_meta)
        if info:
            print("Incident saved:", info)

        now = time.time()

        # ---- AUTO TRIGGER ----
        if now - start_time > MIN_BUFFER_TIME:
            if fake_meta["risk_overall"] >= RISK_TRIGGER:
                if now - last_trigger > COOLDOWN:
                    print("[AUTO] Triggering incident...")
                    engine.trigger()
                    last_trigger = now

        # ---- DISPLAY ----
        cv2.imshow("Replay Demo (Press T to Trigger)", frame)
        key = cv2.waitKey(1)

        if key == ord('t'):
            if now - last_trigger > COOLDOWN:
                print("[MANUAL] Triggering incident...")
                engine.trigger()
                last_trigger = now

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("=== Demo Finished ===")


if __name__ == "__main__":
    demo_run()
