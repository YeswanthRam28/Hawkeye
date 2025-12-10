# explainability/pose_overlay.py
import cv2

def _normalize_people_input(people):
    """
    Accept either:
      - single person's keypoints: list of kp dicts => return [that list]
      - list of persons: [ [kp dicts], [kp dicts], ... ]
    """
    if people is None:
        return []
    if not isinstance(people, (list, tuple)):
        return []
    # if first element is dict with 'x' key -> single person list of keypoints
    if len(people) > 0 and isinstance(people[0], dict) and "x" in people[0]:
        return [people]
    # else assume it's a list of persons (possibly empty)
    return people

def draw_pose(frame, people_keypoints, confidence_threshold=0.3):
    """
    Hybrid API:
      people_keypoints may be:
        - single person: list of {"x","y","score"}
        - multi-person: list of such lists
    Returns frame with drawn skeletons (same visual style as before).
    """
    people = _normalize_people_input(people_keypoints)

    for keypoints in people:
        if not keypoints:
            continue
        # draw joints
        for kp in keypoints:
            try:
                x, y, s = int(kp["x"]), int(kp["y"]), float(kp.get("score", 1.0))
            except Exception:
                continue
            if s >= confidence_threshold:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # safe skeleton: connect consecutive keypoints as a simple chain (backward compatible)
        for i in range(len(keypoints) - 1):
            a = keypoints[i]; b = keypoints[i+1]
            if a and b and a.get("score", 0) >= confidence_threshold and b.get("score", 0) >= confidence_threshold:
                x1, y1 = int(a["x"]), int(a["y"])
                x2, y2 = int(b["x"]), int(b["y"])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)

    return frame
