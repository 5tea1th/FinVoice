'use client';

import React from 'react';

/* ─── Stage 0: Ingestion — Audio waveform being captured ─── */
function IngestionIllustration({ color }: { color: string }) {
  const bars = [
    { x: 60, h: 50, dur: '1.2s', delay: '0s' },
    { x: 80, h: 70, dur: '0.9s', delay: '0.15s' },
    { x: 100, h: 90, dur: '1.1s', delay: '0.05s' },
    { x: 120, h: 60, dur: '1.3s', delay: '0.25s' },
    { x: 140, h: 80, dur: '0.8s', delay: '0.1s' },
    { x: 160, h: 55, dur: '1.0s', delay: '0.3s' },
    { x: 180, h: 75, dur: '1.15s', delay: '0.2s' },
    { x: 200, h: 45, dur: '0.95s', delay: '0.35s' },
    { x: 220, h: 65, dur: '1.25s', delay: '0.08s' },
    { x: 240, h: 40, dur: '1.05s', delay: '0.18s' },
  ];

  return (
    <svg viewBox="0 0 300 180" className="ps-illustration">
      <defs>
        <linearGradient id="ig-bar" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.9" />
          <stop offset="100%" stopColor={color} stopOpacity="0.25" />
        </linearGradient>
        <filter id="ig-glow"><feGaussianBlur stdDeviation="4" /></filter>
      </defs>

      <ellipse cx="150" cy="90" rx="110" ry="50" fill={color} opacity="0.12" filter="url(#ig-glow)" />

      {/* Microphone icon */}
      <g transform="translate(18, 55)" opacity="0.7">
        <rect x="0" y="0" width="16" height="28" rx="8" fill="none" stroke={color} strokeWidth="2" />
        <path d="M-6,22 C-6,38 22,38 22,22" fill="none" stroke={color} strokeWidth="2" />
        <line x1="8" y1="38" x2="8" y2="48" stroke={color} strokeWidth="2" />
        <line x1="2" y1="48" x2="14" y2="48" stroke={color} strokeWidth="1.5" />
      </g>

      {/* Equalizer bars */}
      {bars.map((bar, j) => (
        <rect
          key={j}
          x={bar.x}
          y={90 - bar.h / 2}
          width="10"
          rx="5"
          height={bar.h}
          fill="url(#ig-bar)"
        >
          <animate
            attributeName="height"
            values={`${bar.h * 0.3};${bar.h};${bar.h * 0.5};${bar.h * 0.8};${bar.h * 0.3}`}
            dur={bar.dur}
            begin={bar.delay}
            repeatCount="indefinite"
          />
          <animate
            attributeName="y"
            values={`${90 - bar.h * 0.15};${90 - bar.h / 2};${90 - bar.h * 0.25};${90 - bar.h * 0.4};${90 - bar.h * 0.15}`}
            dur={bar.dur}
            begin={bar.delay}
            repeatCount="indefinite"
          />
        </rect>
      ))}

      {/* Incoming sound arcs */}
      {[0, 1, 2].map(k => (
        <path
          key={k}
          d={`M${268 + k * 10},65 A${12 + k * 8},${22 + k * 10} 0 0,1 ${268 + k * 10},115`}
          fill="none"
          stroke={color}
          strokeWidth="1.8"
          strokeLinecap="round"
          opacity="0"
        >
          <animate attributeName="opacity" values="0;0.8;0" dur="2s" begin={`${k * 0.5}s`} repeatCount="indefinite" />
        </path>
      ))}
    </svg>
  );
}

/* ─── Stage 1: Transcription — Text materializing from audio ─── */
function TranscriptionIllustration({ color }: { color: string }) {
  const lines = [
    { y: 30, w: 180, delay: '0s' },
    { y: 50, w: 140, delay: '0.6s' },
    { y: 70, w: 165, delay: '1.2s' },
    { y: 90, w: 120, delay: '1.8s' },
    { y: 110, w: 155, delay: '2.4s' },
    { y: 130, w: 90, delay: '3.0s' },
  ];

  return (
    <svg viewBox="0 0 300 180" className="ps-illustration">
      <defs>
        <linearGradient id="tr-fade" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={color} stopOpacity="0.8" />
          <stop offset="70%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Document outline */}
      <rect x="30" y="12" width="220" height="156" rx="8" fill="none" stroke={color} strokeWidth="1" opacity="0.25" />

      {/* Typing cursor */}
      <rect x="52" y="28" width="2.5" height="14" fill={color}>
        <animate attributeName="opacity" values="1;0.2;1" dur="1s" repeatCount="indefinite" />
      </rect>

      {/* Text lines with typewriter */}
      {lines.map((line, j) => (
        <g key={j}>
          <rect x="55" y={line.y} width={line.w} height="6" rx="3" fill="url(#tr-fade)">
            <animate attributeName="width" values={`0;${line.w}`} dur="2s" begin={line.delay} repeatCount="indefinite" />
            <animate attributeName="opacity" values="0;1;1;0" dur="4s" begin={line.delay} repeatCount="indefinite" />
          </rect>
          {/* Speaker markers */}
          {j % 2 === 0 ? (
            <circle cx="45" cy={line.y + 3} r="3.5" fill="none" stroke={color} strokeWidth="1.2" opacity="0">
              <animate attributeName="opacity" values="0;0.7;0.7;0" dur="4s" begin={line.delay} repeatCount="indefinite" />
            </circle>
          ) : (
            <rect x="41" y={line.y} width="7" height="6" rx="1.5" fill="none" stroke={color} strokeWidth="1.2" opacity="0">
              <animate attributeName="opacity" values="0;0.7;0.7;0" dur="4s" begin={line.delay} repeatCount="indefinite" />
            </rect>
          )}
        </g>
      ))}

      {/* Sound wave entering from left */}
      <g>
        <path d="M8,70 Q15,48 8,26" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round">
          <animate attributeName="opacity" values="0.4;0.8;0.4" dur="2.5s" repeatCount="indefinite" />
        </path>
        <path d="M15,80 Q24,48 15,16" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round">
          <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.5s" begin="0.3s" repeatCount="indefinite" />
        </path>
      </g>
    </svg>
  );
}

/* ─── Stage 2: NLU & Extraction — Entities on a clean document ─── */
function NLUExtractionIllustration({ color }: { color: string }) {
  const COL1 = 28;
  const COL2 = 156;
  const ROW_GAP = 48;
  const ROW_START = 24;
  const BOX_W = 100;
  const BOX_H = 28;

  const entities = [
    { col: COL1, row: 0, label: '₹ 24,500',   tag: 'AMOUNT',  tagColor: '#f59e0b' },
    { col: COL2, row: 0, label: '****4821',    tag: 'ACCOUNT', tagColor: '#6366f1' },
    { col: COL1, row: 1, label: '12 Jan 2025', tag: 'DATE',    tagColor: '#10b981' },
    { col: COL2, row: 1, label: 'Rahul Mehta', tag: 'NAME',    tagColor: '#ef4444' },
    { col: COL1, row: 2, label: 'Balance Inq', tag: 'INTENT',  tagColor: '#a855f7' },
    { col: COL2, row: 2, label: 'Savings Acct', tag: 'PRODUCT', tagColor: '#f59e0b' },
  ];

  return (
    <svg viewBox="0 0 300 180" className="ps-illustration">
      {/* Document background */}
      <rect x="16" y="8" width="268" height="164" rx="10" fill="none" stroke={color} strokeWidth="0.8" opacity="0.15" />

      {/* Scanning beam */}
      <line x1="20" y1="0" x2="20" y2="180" stroke={color} strokeWidth="2" opacity="0.06">
        <animate attributeName="x1" values="20;280;20" dur="4s" repeatCount="indefinite" />
        <animate attributeName="x2" values="20;280;20" dur="4s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.04;0.2;0.04" dur="4s" repeatCount="indefinite" />
      </line>

      {/* Entity grid */}
      {entities.map((ent, j) => {
        const y = ROW_START + ent.row * ROW_GAP;
        const delay = `${j * 0.6}s`;
        return (
          <g key={j}>
            {/* Box fill */}
            <rect x={ent.col} y={y} width={BOX_W} height={BOX_H} rx="6" fill={ent.tagColor} opacity="0">
              <animate attributeName="opacity" values="0;0.15;0.15;0" dur="4.5s" begin={delay} repeatCount="indefinite" />
            </rect>
            {/* Box border */}
            <rect x={ent.col} y={y} width={BOX_W} height={BOX_H} rx="6" fill="none" stroke={ent.tagColor} strokeWidth="1.5" opacity="0">
              <animate attributeName="opacity" values="0;0.8;0.8;0" dur="4.5s" begin={delay} repeatCount="indefinite" />
            </rect>
            {/* Value text */}
            <text x={ent.col + BOX_W / 2} y={y + BOX_H / 2 + 1} textAnchor="middle" dominantBaseline="middle"
              fontSize="9.5" fontFamily="monospace" fill={ent.tagColor} opacity="0">
              {ent.label}
              <animate attributeName="opacity" values="0;0.9;0.9;0" dur="4.5s" begin={delay} repeatCount="indefinite" />
            </text>
            {/* Tag label */}
            <text x={ent.col + BOX_W / 2} y={y + BOX_H + 13} textAnchor="middle"
              fontSize="7.5" fontFamily="monospace" fontWeight="bold" letterSpacing="0.1em" fill={ent.tagColor} opacity="0">
              {ent.tag}
              <animate attributeName="opacity" values="0;0.65;0.65;0" dur="4.5s" begin={delay} repeatCount="indefinite" />
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ─── Stage 3: Compliance & Fraud — Shield scanning with checkmarks ─── */
function ComplianceIllustration({ color }: { color: string }) {
  const checkItems = [
    { y: 28, pass: true, label: 'KYC Verified', delay: '0s' },
    { y: 52, pass: true, label: 'Consent Given', delay: '0.8s' },
    { y: 76, pass: false, label: 'Risk Threshold', delay: '1.6s' },
    { y: 100, pass: true, label: 'Disclosure OK', delay: '2.4s' },
    { y: 124, pass: true, label: 'No Fraud Signal', delay: '3.2s' },
  ];

  return (
    <svg viewBox="0 0 300 180" className="ps-illustration">
      {/* Large shield */}
      <g transform="translate(225, 90)">
        <path d="M0,-50 L35,-30 V10 C35,40 0,55 0,55 C0,55 -35,40 -35,10 V-30 Z" fill={color} opacity="0.08" />
        <path d="M0,-50 L35,-30 V10 C35,40 0,55 0,55 C0,55 -35,40 -35,10 V-30 Z" fill="none" stroke={color} strokeWidth="1.8">
          <animate attributeName="opacity" values="0.4;0.7;0.4" dur="3s" repeatCount="indefinite" />
        </path>
        {/* Checkmark inside shield */}
        <path d="M-10,2 L-4,8 L12,-8" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" opacity="0.5">
          <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3s" repeatCount="indefinite" />
        </path>
      </g>

      {/* Pulse ring */}
      <circle cx="225" cy="90" r="30" fill="none" stroke={color} strokeWidth="1" opacity="0">
        <animate attributeName="r" values="30;65;65" dur="3s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.5;0;0" dur="3s" repeatCount="indefinite" />
      </circle>

      {/* Checklist */}
      {checkItems.map((item, j) => {
        const itemColor = item.pass ? '#10b981' : '#ef4444';
        return (
          <g key={j}>
            <rect x="30" y={item.y} width="165" height="18" rx="4" fill={itemColor} opacity="0">
              <animate attributeName="opacity" values="0;0.06;0.06;0" dur="5s" begin={item.delay} repeatCount="indefinite" />
            </rect>

            {/* Icon */}
            <g transform={`translate(42, ${item.y + 9})`} opacity="0">
              <circle cx="0" cy="0" r="6" fill="none" stroke={itemColor} strokeWidth="1.5" />
              {item.pass
                ? <path d="M-3,0 L-1,2.5 L3,-2.5" fill="none" stroke={itemColor} strokeWidth="1.5" strokeLinecap="round" />
                : <path d="M-2.5,-2.5 L2.5,2.5 M2.5,-2.5 L-2.5,2.5" stroke={itemColor} strokeWidth="1.5" strokeLinecap="round" />
              }
              <animate attributeName="opacity" values="0;0.9;0.9;0" dur="5s" begin={item.delay} repeatCount="indefinite" />
            </g>

            {/* Label */}
            <text x="56" y={item.y + 13} fontSize="9" fontFamily="var(--font-body), system-ui" fill={itemColor} opacity="0">
              {item.label}
              <animate attributeName="opacity" values="0;0.8;0.8;0" dur="5s" begin={item.delay} repeatCount="indefinite" />
            </text>

            {/* Progress bar */}
            <rect x="140" y={item.y + 6} width="50" height="5" rx="2.5" fill="white" opacity="0.06" />
            <rect x="140" y={item.y + 6} width="0" height="5" rx="2.5" fill={itemColor} opacity="0.5">
              <animate attributeName="width" values={`0;${item.pass ? 50 : 30}`} dur="1.5s" begin={item.delay} repeatCount="indefinite" />
            </rect>
          </g>
        );
      })}
    </svg>
  );
}

/* ─── Stage 4: Output & Storage — Data flowing into structured storage ─── */
function OutputStorageIllustration({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 300 180" className="ps-illustration">
      {/* JSON structure */}
      <g opacity="0.5" transform="translate(30, 18)">
        <text x="0" y="14" fontSize="12" fontFamily="monospace" fill={color}>{'{'}</text>
        <text x="14" y="34" fontSize="9.5" fontFamily="monospace" fill={color} opacity="0">
          &quot;transcript&quot;: &quot;...&quot;
          <animate attributeName="opacity" values="0;0.9;0.9;0" dur="6s" begin="0s" repeatCount="indefinite" />
        </text>
        <text x="14" y="52" fontSize="9.5" fontFamily="monospace" fill={color} opacity="0">
          &quot;entities&quot;: [14]
          <animate attributeName="opacity" values="0;0.9;0.9;0" dur="6s" begin="0.8s" repeatCount="indefinite" />
        </text>
        <text x="14" y="70" fontSize="9.5" fontFamily="monospace" fill={color} opacity="0">
          &quot;compliance&quot;: 0.87
          <animate attributeName="opacity" values="0;0.9;0.9;0" dur="6s" begin="1.6s" repeatCount="indefinite" />
        </text>
        <text x="14" y="88" fontSize="9.5" fontFamily="monospace" fill={color} opacity="0">
          &quot;fraud_flag&quot;: false
          <animate attributeName="opacity" values="0;0.9;0.9;0" dur="6s" begin="2.4s" repeatCount="indefinite" />
        </text>
        <text x="14" y="106" fontSize="9.5" fontFamily="monospace" fill={color} opacity="0">
          &quot;speakers&quot;: [2]
          <animate attributeName="opacity" values="0;0.9;0.9;0" dur="6s" begin="3.2s" repeatCount="indefinite" />
        </text>
        <text x="0" y="124" fontSize="12" fontFamily="monospace" fill={color}>{'}'}</text>
      </g>

      {/* Arrow flowing to database */}
      <g opacity="0.5">
        <path d="M190,90 L222,90" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeDasharray="5 5">
          <animate attributeName="stroke-dashoffset" values="0;-20" dur="1.5s" repeatCount="indefinite" />
        </path>
        <polygon points="220,85 230,90 220,95" fill={color} opacity="0.6">
          <animate attributeName="opacity" values="0.4;0.8;0.4" dur="1.5s" repeatCount="indefinite" />
        </polygon>
      </g>

      {/* Database cylinder */}
      <g transform="translate(240, 50)">
        <ellipse cx="28" cy="0" rx="28" ry="12" fill={color} stroke={color} strokeWidth="1.5" opacity="0.6" />
        <rect x="0" y="0" width="56" height="68" fill={color} opacity="0.05" />
        <line x1="0" y1="0" x2="0" y2="68" stroke={color} strokeWidth="1.5" opacity="0.5" />
        <line x1="56" y1="0" x2="56" y2="68" stroke={color} strokeWidth="1.5" opacity="0.5" />
        <ellipse cx="28" cy="68" rx="28" ry="12" fill={color} stroke={color} strokeWidth="1.5" opacity="0.5" />
        <ellipse cx="28" cy="23" rx="28" ry="12" fill="none" stroke={color} strokeWidth="0.8" opacity="0.35" />
        <ellipse cx="28" cy="46" rx="28" ry="12" fill="none" stroke={color} strokeWidth="0.8" opacity="0.35" />
        {/* Pulse */}
        <ellipse cx="28" cy="34" rx="22" ry="10" fill={color} opacity="0">
          <animate attributeName="opacity" values="0;0.2;0" dur="2.5s" repeatCount="indefinite" />
        </ellipse>
      </g>

      {/* Data particles */}
      {[0, 1, 2, 3].map(k => (
        <circle key={k} cx="190" cy="90" r="2.5" fill={color} opacity="0">
          <animate attributeName="cx" values="190;245" dur="1.8s" begin={`${k * 0.45}s`} repeatCount="indefinite" />
          <animate attributeName="cy" values={`${82 + k * 5};${68 + k * 12}`} dur="1.8s" begin={`${k * 0.45}s`} repeatCount="indefinite" />
          <animate attributeName="opacity" values="0;0.8;0.8;0" dur="1.8s" begin={`${k * 0.45}s`} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
}

/* ─── Exports ─── */
const ILLUSTRATIONS = [
  IngestionIllustration,
  TranscriptionIllustration,
  NLUExtractionIllustration,
  ComplianceIllustration,
  OutputStorageIllustration,
];

export default function StageIllustration({ stage, color }: { stage: number; color: string }) {
  const Comp = ILLUSTRATIONS[stage];
  if (!Comp) return null;
  return <Comp color={color} />;
}
