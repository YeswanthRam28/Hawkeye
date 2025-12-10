
# 🦅 Hawkeye: Advanced Multi-Modal Surveillance Platform

> **The Event Fusion Engine.** Integrating Vision, Audio, and Motion into a single intelligent risk-assessment dashboard.

![Hawkeye Dashboard](https://via.placeholder.com/800x450.png?text=Hawkeye+Dashboard+Use+Screenshot+Here)

## 🌟 Overview

**Hawkeye** is a scalable surveillance platform designed for the **[OSPC Hackathon]**. It solves the problem of disjointed security systems by fusing three critical data streams:

1.  **Vision Intelligence**: Object detection (Weapons, Persons), Pose Estimation (Falls, Fights).
2.  **Audio Intelligence**: Anomaly detection (Screams, Gunshots, Explosions).
3.  **Motion & Crowd Dynamics**: Panic detection, Crowd Surges, Abnormal Velocity.

This repository serves as the **Platform Core (Member 6)**, aggregating signals from all specialized engines into a unified "Fusion & Risk Engine" and presenting them on a "Judge-Pleasing" Real-Time Dashboard.

---

## 🛠️ Tech Stack

-   **Frontend**: React.js, Chart.js, Glassmorphism UI
-   **Backend**: Python FastAPI, WebSockets, AsyncIO
-   **Communication**: REST API + WebSocket realtime streams
-   **Architecture**: Modular "Micro-Service-Like" connection to sub-systems

---

## 🚀 Quick Start (Local Demo)

### 1. Prerequisites
-   Python 3.10+
-   Node.js 16+
-   Windows/Linux/Mac

### 2. Backend Setup
```bash
# Activate Virtual Environment (Windows)
.\henv\Scripts\Activate.ps1

# Run the Fusion Engine Server
uvicorn backend.main:app --reload
```
*Port: `8000` (API & WebSocket)*

### 3. Frontend Setup
```bash
cd frontend
npm install
npm start
```
*Port: `3000` (Dashboard Application)*

---

## 🧩 Architecture & Roles

This repository is optimized for the **6-Member Team Workflow**:

| Role | Responsibility | Endpoint Integration |
| :--- | :--- | :--- |
| **Member 1 (Vision)** | YOLOv8 / MediaPipe | `/vision/frame-analysis` |
| **Member 2 (Audio)** | Audio Classifier | `/audio/audio-analysis` |
| **Member 3 (Motion)** | Optical Flow | `/motion/crowd-analysis` |
| **Member 4 (Fusion)** | Risk Logic | `/risk/status` (Implemented here) |
| **Member 5 (Forensics)** | Explainability | `/replay/meta` |
| **Member 6 (Platform)** | **This Dashboard** | **UI/UX Core** |

---

## 🔌 Integration

To connect real AI models to this platform, see **[INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)**.
It supports:
1.  **Direct HTTP Linking** (Run your model on port 8001, 8002...)
2.  **Mock Simulation** (Pre-loaded for presentation safety)

---

## ⚠️ Hackathon Notes

-   **Default Admin**: `admin` / `password`
-   **Mock Mode**: Enabled by default in `backend/config.py` to prevent demo crashes.
-   **Latency**: optimized for <100ms updates via WebSockets.
