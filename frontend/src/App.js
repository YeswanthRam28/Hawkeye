import React, { useEffect, useRef, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

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
    img.src = "data:image/jpeg;base64," + imageBase64;
    // after image loads, overlay drawing occurs in onload, so we schedule draw after small timeout
    setTimeout(() => {
      // draw boxes
      ctx.lineWidth = 2;
      ctx.font = "18px Arial";
      // bounding boxes
      (overlays.boxes || []).forEach((b) => {
        ctx.strokeStyle = "lime";
        ctx.strokeRect(b.x * (canvas.width / 1280), b.y * (canvas.height / 720), b.w * (canvas.width / 1280), b.h * (canvas.height / 720));
        ctx.fillStyle = "lime";
        ctx.fillText(`${b.label} ${b.conf}`, b.x * (canvas.width / 1280), Math.max(20, b.y * (canvas.height / 720)));
      });
      // skeleton keypoints
      (overlays.skeletons || []).forEach((s) => {
        ctx.strokeStyle = "cyan";
        ctx.beginPath();
        s.keypoints.forEach((p, idx) => {
          const x = p.x * (canvas.width / 1280);
          const y = p.y * (canvas.height / 720);
          ctx.fillRect(x - 3, y - 3, 6, 6);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
      // simple heatmap placeholder as translucent rectangles
      const hm = overlays.heatmap || [];
      const rows = hm.length;
      const cols = hm[0]?.length || 0;
      const cellW = canvas.width / cols / 2; // small visual
      const cellH = canvas.height / rows / 6;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const val = hm[r][c];
          const alpha = Math.min(0.7, Math.max(0.05, val));
          ctx.fillStyle = `rgba(255,0,0,${alpha})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    }, 25);
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

  // chart data
  const chartData = {
    labels: riskHistory.map((_, i) => i),
    datasets: [
      {
        label: "Risk Score",
        data: riskHistory,
        tension: 0.2,
      },
    ],
  };

  return (
    <div style={{ padding: 12, fontFamily: "Arial, sans-serif" }}>
      <h2>Hawkeye — Live Dashboard (Frontend)</h2>
      <div style={{ display: "flex", gap: 12 }}>
        <div>
          <canvas
            ref={canvasRef}
            width={960}
            height={540}
            style={{ border: "1px solid #333", background: "#000" }}
          />
          <div style={{ marginTop: 8 }}>
            <button onClick={() => requestReplay(1.0)}>Replay 1x</button>{" "}
            <button onClick={() => requestReplay(0.5)}>Replay 0.5x</button>{" "}
            <button onClick={() => requestReplay(2.0)}>Replay 2x</button>{" "}
            <button onClick={exportClip}>Export Clip</button>
          </div>
          <div style={{ marginTop: 8 }}>
            <strong>WS:</strong> {wsState} — <strong>Frames:</strong> {framesReceived}
          </div>
          {latestExportFile && (
            <div style={{ marginTop: 8 }}>
              Latest export: <code>{latestExportFile}</code>
            </div>
          )}
        </div>

        <div style={{ width: 360 }}>
          <div style={{ marginBottom: 12 }}>
            <h4>Risk Timeline</h4>
            <Line data={chartData} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <h4>Replay Status</h4>
            <div>{replayStatus || "idle"}</div>
          </div>

          <div>
            <h4>Quick Debug</h4>
            <div>Token: {token ? token.slice(0, 24) + "..." : "loading"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
