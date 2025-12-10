
import React, { useEffect, useRef, useState } from 'react';
import RiskChart from './RiskChart';
import AlertsPanel from './AlertsPanel';

const BACKEND = "http://127.0.0.1:8000";

const Dashboard = ({ token, wsState, canvasRef, handleReplayRequest, handleExport, framesReceived, riskHistory, latestExportFile, replayStatus }) => {

    // Polling generic Risk Status for the header stats
    const [stats, setStats] = useState({ risk: 0, contributors: {} });

    useEffect(() => {
        const i = setInterval(() => {
            fetch(`${BACKEND}/risk/status`)
                .then(r => r.json())
                .then(d => {
                    setStats({
                        risk: d.overall_risk_score,
                        contributors: d.contributors
                    });
                })
                .catch(() => { });
        }, 500);
        return () => clearInterval(i);
    }, []);

    const isHighRisk = stats.risk > 0.7;

    return (
        <div style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 1fr)',
            gap: '20px',
            height: 'calc(100vh - 40px)',
            padding: '20px'
        }}>

            {/* Left Column: Video Main + KPI Bar */}
            <div style={{ display: 'grid', gridTemplateRows: 'auto 1fr auto', gap: '20px' }}>

                {/* Header Stats Bar */}
                <div className="glass-panel" style={{ padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <div style={{ background: '#3b82f6', borderRadius: '50%', width: '12px', height: '12px' }} />
                        <h2 style={{ margin: 0, fontSize: '1.25rem' }}>HAWKEYE <span style={{ fontWeight: 400, opacity: 0.7 }}>INTELLIGENCE</span></h2>
                    </div>

                    <div style={{ display: 'flex', gap: '24px' }}>
                        <StatItem label="CROWD DENSITY" value={stats.contributors.motion ? Math.round(stats.contributors.motion * 100) + '%' : '--'} />
                        <StatItem label="AUDIO ANOMALY" value={stats.contributors.audio ? Math.round(stats.contributors.audio * 100) + '%' : '--'} />
                        <StatItem label="THREAT LEVEL" value={Math.round(stats.risk * 100) + '%'} color={isHighRisk ? '#ef4444' : '#22c55e'} />
                    </div>
                </div>

                {/* Main Video Area */}
                <div className={`glass-panel ${isHighRisk ? 'risk-alert-glow' : ''}`} style={{ padding: '4px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
                    <canvas
                        ref={canvasRef}
                        width={1280}
                        height={720}
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                </div>

                {/* Actions Bar */}
                <div className="glass-panel" style={{ padding: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.85rem', color: '#64748b' }}>STATUS: {wsState.toUpperCase()}</span>
                        <span style={{ fontSize: '0.85rem', color: '#64748b' }}>•</span>
                        <span style={{ fontSize: '0.85rem', color: '#64748b' }}>FRAMES: {framesReceived}</span>
                    </div>

                    <div style={{ display: 'flex', gap: '12px' }}>
                        {replayStatus === 'playing' ? (
                            <button className="btn-primary" style={{ background: '#f59e0b' }}>REPLAYING...</button>
                        ) : (
                            <>
                                <button className="btn-primary" onClick={() => handleReplayRequest(0.5)}>FAILED SLOW-MO</button>
                                <button className="btn-primary" onClick={() => handleReplayRequest(1.0)}>INSTANT REPLAY</button>
                            </>
                        )}
                        <button className="btn-primary" style={{ background: '#ef4444' }} onClick={handleExport}>EXPORT EVIDENCE</button>
                    </div>
                </div>
            </div>

            {/* Right Column: Risk Graph + Alerts */}
            <div style={{ display: 'grid', gridTemplateRows: '1fr 2fr', gap: '20px' }}>

                {/* Risk Graph Panel */}
                <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ margin: '0 0 10px 0', fontSize: '1rem', color: '#94a3b8' }}>RISK TIMELINE</h3>
                    <div style={{ flex: 1, minHeight: 0 }}>
                        <RiskChart riskHistory={riskHistory} />
                    </div>
                </div>

                {/* Alerts / Events Panel */}
                <AlertsPanel />

                {latestExportFile && (
                    <div className="glass-panel" style={{ padding: '12px', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                        <div style={{ fontSize: '0.85rem', color: '#fff' }}>Evidence Saved:</div>
                        <code style={{ fontSize: '0.75rem', color: '#86efac' }}>{latestExportFile}</code>
                    </div>
                )}

            </div>
        </div>
    );
};

const StatItem = ({ label, value, color = '#fff' }) => (
    <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '0.7rem', color: '#94a3b8', letterSpacing: '0.05em' }}>{label}</div>
        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: color, lineHeight: 1 }}>{value}</div>
    </div>
);

export default Dashboard;
