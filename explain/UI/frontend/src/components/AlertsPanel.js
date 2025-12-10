
import React, { useEffect, useState } from 'react';

const BACKEND = "http://127.0.0.1:8000";

const AlertsPanel = ({ replayMode = false }) => {
    // We poll for events from Vision, Audio, Motion
    const [events, setEvents] = useState([]);

    useEffect(() => {
        if (replayMode) return; // In replay mode, events might come from replay stream (not handled here for simplicity)

        const fetchEvents = async () => {
            try {
                // Parallel fetch
                const [v, a, m] = await Promise.all([
                    fetch(`${BACKEND}/vision/events`).then(r => r.json()),
                    fetch(`${BACKEND}/audio/events`).then(r => r.json()),
                    fetch(`${BACKEND}/motion/events`).then(r => r.json())
                ]);

                // Combine and sort by confidence
                const all = [
                    ...v.events.map(e => ({ ...e, source: 'VISION', ts: v.timestamp })),
                    ...a.events.map(e => ({ ...e, source: 'AUDIO', ts: a.timestamp })),
                    ...m.events.map(e => ({ ...e, source: 'MOTION', ts: m.timestamp }))
                ];

                // Filter low confidence to reduce noise
                const highConf = all.filter(e => e.confidence > 0.7);

                // Add to list (keep last 10)
                setEvents(prev => {
                    // avoid duplicates if timestamp is same? 
                    // simple robust approach: just prepend new ones that look unique or just replace 'recent' list
                    // For a log: prepend new items.
                    // For a "Live State" panel: just show what is happening NOW.

                    // Let's implement a "Log" style.
                    const newEvents = highConf.filter(n =>
                        !prev.some(p => p.type === n.type && Math.abs(p.ts - n.ts) < 0.1)
                    );

                    if (newEvents.length === 0) return prev;
                    return [...newEvents, ...prev].slice(0, 15);
                });

            } catch (e) {
                // ignore poll errors
            }
        };

        const interval = setInterval(fetchEvents, 800);
        return () => clearInterval(interval);
    }, [replayMode]);

    return (
        <div className="glass-panel" style={{ padding: '16px', height: '100%', overflowY: 'auto' }}>
            <h3 style={{ marginTop: 0, marginBottom: '12px', fontSize: '1.1rem', color: '#fff', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '8px' }}>
                Live Alerts Log
            </h3>

            {events.length === 0 && (
                <div style={{ color: '#64748b', fontStyle: 'italic', textAlign: 'center', marginTop: '20px' }}>
                    System Secure. No anomalies.
                </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {events.map((evt, i) => (
                    <div key={i} style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '8px',
                        background: 'rgba(255,255,255,0.03)',
                        borderRadius: '6px',
                        borderLeft: `3px solid ${getSeverityColor(evt.confidence)}`
                    }}>
                        <div>
                            <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{evt.type.replace('_', ' ').toUpperCase()}</div>
                            <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                                {evt.source} • {new Date(evt.ts * 1000).toLocaleTimeString()}
                            </div>
                        </div>
                        <div className={`tag ${evt.confidence > 0.9 ? 'high' : 'medium'}`}>
                            {Math.round(evt.confidence * 100)}%
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

function getSeverityColor(conf) {
    if (conf > 0.9) return '#ef4444'; // Red
    if (conf > 0.8) return '#f59e0b'; // Amber
    return '#3b82f6'; // Blue
}

export default AlertsPanel;
