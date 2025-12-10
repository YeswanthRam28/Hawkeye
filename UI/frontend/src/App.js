import React, { useEffect, useRef, useState } from "react";
import Dashboard from "./components/Dashboard";

const BACKEND = "http://127.0.0.1:8000"; // backend base

function App() {
  const [token, setToken] = useState(null);
  const [wsState, setWsState] = useState("idle"); // idle|connecting|open|closed
  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null); // for decoding images
  const [framesReceived, setFramesReceived] = useState(0);
  const [riskHistory, setRiskHistory] = useState([]); // latest 50 risk values
  const riskMaxLen = 100;
  const [replayStatus, setReplayStatus] = useState(null);
  const [latestExportFile, setLatestExportFile] = useState(null);

  // get token (demo admin/password)
  useEffect(() => {
    async function getToken() {
      try {
        const res = await fetch(BACKEND + "/token", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: "admin", password: "password" }),
        });
        const j = await res.json();
        setToken(j.access_token);
      } catch (e) {
        console.error("token fetch error", e);
      }
    }
    getToken();
  }, []);

  // open websocket whenever token available
  useEffect(() => {
    if (!token) return;
    setWsState("connecting");
    const ws = new WebSocket(`ws://127.0.0.1:8000/stream/video?token=${token}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WS open");
      setWsState("open");
    };

    ws.onmessage = (ev) => {
      try {
        const pkt = JSON.parse(ev.data);
        handlePacket(pkt);
      } catch (err) {
        console.warn("non-json ws", ev.data);
      }
    };

    ws.onerror = (e) => {
      console.error("WS error", e);
    };

    ws.onclose = (e) => {
      console.log("WS closed", e);
      setWsState("closed");
    };

    return () => {
      ws.close();
    };
  }, [token]);

  // draw image + overlays
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    // prepare offscreen image object
    imgRef.current = new Image();
    imgRef.current.onload = function () {
      // draw at canvas size
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imgRef.current, 0, 0, canvas.width, canvas.height);
    };
  }, []);

  function drawOverlaysOnCanvas(imageBase64, overlays) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;

    // We must handle the case where img is not yet loaded if we reuse the same Object
    // But setting src triggers onload.
    img.onload = () => {
      // Clear and draw Frame
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Draw Overlays
      // Overlays need scaling from 1280x720 to Canvas Size if different
      // In Dashboard.js we set width={1280} height={720}, so 1:1, but CSS might shrink it.
      // Canvas internal resolution is 1280x720.

      // -------------------------
      // PREMIUM OVERLAY STYLES
      // -------------------------
      ctx.lineWidth = 2;
      ctx.font = "bold 16px Inter, sans-serif";

      (overlays.boxes || []).forEach((b) => {
        const isTarget = ['knife', 'gun', 'person'].includes(b.label);
        const color = b.label === 'person' ? '#3b82f6' : '#ef4444';

        ctx.strokeStyle = color;
        ctx.strokeRect(b.x, b.y, b.w, b.h);

        // Label tag background
        ctx.fillStyle = color;
        const text = `${b.label} ${Math.round(b.conf * 100)}%`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(b.x, b.y - 24, textWidth + 12, 24);

        ctx.fillStyle = "#fff";
        ctx.fillText(text, b.x + 6, b.y - 6);
      });

      // Skeleton
      (overlays.skeletons || []).forEach((s) => {
        ctx.strokeStyle = "rgba(59, 130, 246, 0.8)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        s.keypoints.forEach((p, idx) => {
          if (idx === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
          // dots
          ctx.fillStyle = "#60a5fa";
          ctx.fillRect(p.x - 3, p.y - 3, 6, 6);
        });
        ctx.stroke();
      });
    }

    img.src = "data:image/jpeg;base64," + imageBase64;
  }

  // very simple audio chunk playback (works for demo)
  function playAudioBase64(wavBase64) {
    if (!wavBase64) return;
    try {
      const binary = atob(wavBase64);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play().catch((e) => {
        // Safari / autoplay restrictions may block; ignore for demo
      });
      setTimeout(() => URL.revokeObjectURL(url), 3000);
    } catch (e) {
      console.warn("play audio err", e);
    }
  }

  function handlePacket(pkt) {
    if (!pkt || !pkt.type) return;
    if (pkt.type === "frame") {
      // live frame
      drawOverlaysOnCanvas(pkt.image_b64, pkt.overlays || {});
      setFramesReceived((s) => s + 1);
      if (typeof pkt.risk_score === "number") {
        setRiskHistory((h) => {
          const next = [...h, pkt.risk_score];
          if (next.length > riskMaxLen) next.shift();
          return next;
        });
      }
      if (pkt.audio_b64) playAudioBase64(pkt.audio_b64);
    } else if (pkt.type === "replay_frame") {
      // replay frame (play and draw)
      drawOverlaysOnCanvas(pkt.image_b64, pkt.overlays || {});
      if (pkt.audio_b64) playAudioBase64(pkt.audio_b64);
    } else if (pkt.type === "replay_end") {
      setReplayStatus(pkt.reason || "completed");
    } else {
      // other types: pong/info
    }
  }

  // request replay
  async function requestReplay(speed = 1.0) {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: "replay", speed }));
    setReplayStatus("playing");
  }

  // export clip (calls backend)
  async function exportClip() {
    if (!token) return;
    try {
      const res = await fetch(BACKEND + "/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });
      const j = await res.json();
      if (j.file) {
        setLatestExportFile(j.file);
        // open in new tab
        window.open(`${BACKEND}/exports/${encodeURIComponent(j.file)}?token=${token}`, "_blank");
      } else {
        alert("Export failed");
      }
    } catch (e) {
      console.error("export err", e);
      alert("Export error - see console");
    }
  }

  if (!token) return <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', background: '#0f172a' }}>Authenticating...</div>;

  return (
    <Dashboard
      token={token}
      wsState={wsState}
      canvasRef={canvasRef}
      handleReplayRequest={requestReplay}
      handleExport={exportClip}
      framesReceived={framesReceived}
      riskHistory={riskHistory}
      latestExportFile={latestExportFile}
      replayStatus={replayStatus}
    />
  );
}

export default App;
