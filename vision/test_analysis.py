import base64
import requests

# Encode image file as base64
with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image_b64": img_b64,
    "include_pose": True
}

res = requests.post("http://127.0.0.1:9000/vision/frame-analysis", json=payload)
print("Status:", res.status_code)
print("Response JSON:")
print(res.json())
