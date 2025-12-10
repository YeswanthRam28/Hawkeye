
import React, { useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler,
    Legend,
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler,
    Legend
);

const RiskChart = ({ riskHistory, maxLen = 50 }) => {

    const data = useMemo(() => {
        // Generate simple labels like "T-10s"
        const labels = riskHistory.map((_, i) => {
            const diff = riskHistory.length - 1 - i;
            return diff === 0 ? 'Now' : `-${diff}`;
        });

        return {
            labels,
            datasets: [
                {
                    label: 'Composite Risk Score',
                    data: riskHistory,
                    fill: true,
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderColor: '#ef4444',
                    tension: 0.4, // Smooth curve
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    borderWidth: 2,
                },
            ],
        };
    }, [riskHistory]);

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 }, // critical for performance
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#1e293b',
                titleColor: '#f8fafc',
                bodyColor: '#cbd5e1',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
            }
        },
        scales: {
            x: {
                display: false, // Clean look
                grid: { display: false }
            },
            y: {
                min: 0,
                max: 1,
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                },
                ticks: {
                    color: '#64748b',
                    font: { size: 10 }
                }
            }
        },
    };

    return (
        <div style={{ height: '100%', width: '100%' }}>
            <Line data={data} options={options} />
        </div>
    );
};

export default RiskChart;
