'use client';

import React from 'react';

// ═══════════════════════════════════════════════
//  FEATURES SECTION ICONS
// ═══════════════════════════════════════════════

/* 1 · Compliance — Animated balance scale that tilts */
export function ComplianceIcon({ color = '#ef4444', size = 56 }: { color?: string; size?: number }) {
  const id = `comp-${Math.random().toString(36).slice(2, 6)}`;
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      <defs>
        <linearGradient id={`${id}-g`} x1="0" y1="0" x2="56" y2="56">
          <stop offset="0%" stopColor={color} stopOpacity="0.9" />
          <stop offset="100%" stopColor={color} stopOpacity="0.4" />
        </linearGradient>
      </defs>
      {/* Base */}
      <path d="M22 50 L28 46 L34 50" fill={color} opacity="0.25" />
      <rect x="20" y="50" width="16" height="2.5" rx="1.25" fill={color} opacity="0.35" />
      {/* Pillar */}
      <line x1="28" y1="46" x2="28" y2="16" stroke={`url(#${id}-g)`} strokeWidth="2.5" strokeLinecap="round" />
      {/* Fulcrum jewel */}
      <circle cx="28" cy="14" r="3.5" fill={color} opacity="0.85">
        <animate attributeName="opacity" values="0.7;1;0.7" dur="2.5s" repeatCount="indefinite" />
      </circle>
      {/* Beam — tilts */}
      <g>
        <animateTransform attributeName="transform" type="rotate" values="-5 28 16;5 28 16;-5 28 16" dur="3.5s" repeatCount="indefinite" calcMode="spline" keySplines="0.45 0 0.55 1;0.45 0 0.55 1" />
        <line x1="8" y1="16" x2="48" y2="16" stroke={color} strokeWidth="2" strokeLinecap="round" />
        {/* Left chains */}
        <line x1="8" y1="16" x2="6" y2="26" stroke={color} strokeWidth="1.2" opacity="0.5" />
        <line x1="8" y1="16" x2="16" y2="26" stroke={color} strokeWidth="1.2" opacity="0.5" />
        {/* Left pan */}
        <path d="M3 26 Q11 34 19 26" stroke={color} strokeWidth="1.5" fill={color} fillOpacity="0.1" strokeLinecap="round">
          <animate attributeName="d" values="M3 26 Q11 34 19 26;M3 28 Q11 36 19 28;M3 26 Q11 34 19 26" dur="3.5s" repeatCount="indefinite" calcMode="spline" keySplines="0.45 0 0.55 1;0.45 0 0.55 1" />
        </path>
        {/* Left pan items */}
        <circle cx="9" cy="27" r="2" fill={color} opacity="0.3">
          <animate attributeName="cy" values="27;29;27" dur="3.5s" repeatCount="indefinite" />
        </circle>
        <circle cx="13" cy="26" r="1.5" fill={color} opacity="0.2">
          <animate attributeName="cy" values="26;28;26" dur="3.5s" repeatCount="indefinite" />
        </circle>
        {/* Right chains */}
        <line x1="48" y1="16" x2="40" y2="26" stroke={color} strokeWidth="1.2" opacity="0.5" />
        <line x1="48" y1="16" x2="50" y2="26" stroke={color} strokeWidth="1.2" opacity="0.5" />
        {/* Right pan */}
        <path d="M37 26 Q45 34 53 26" stroke={color} strokeWidth="1.5" fill={color} fillOpacity="0.1" strokeLinecap="round">
          <animate attributeName="d" values="M37 28 Q45 36 53 28;M37 26 Q45 34 53 26;M37 28 Q45 36 53 28" dur="3.5s" repeatCount="indefinite" calcMode="spline" keySplines="0.45 0 0.55 1;0.45 0 0.55 1" />
        </path>
        {/* Right pan items */}
        <circle cx="45" cy="27" r="1.5" fill={color} opacity="0.25">
          <animate attributeName="cy" values="28;26;28" dur="3.5s" repeatCount="indefinite" />
        </circle>
      </g>
    </svg>
  );
}

/* 2 · Fraud Detection — Shield with scanning beam */
export function FraudIcon({ color = '#ef4444', size = 56 }: { color?: string; size?: number }) {
  const id = `fraud-${Math.random().toString(36).slice(2, 6)}`;
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      <defs>
        <linearGradient id={`${id}-fill`} x1="28" y1="4" x2="28" y2="52">
          <stop offset="0%" stopColor={color} stopOpacity="0.2" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
        <linearGradient id={`${id}-scan`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0" />
          <stop offset="40%" stopColor={color} stopOpacity="0.35" />
          <stop offset="60%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
        <clipPath id={`${id}-clip`}>
          <path d="M28 4 L46 14 V30 C46 42 28 52 28 52 C28 52 10 42 10 30 V14 Z" />
        </clipPath>
      </defs>
      {/* Shield */}
      <path d="M28 4 L46 14 V30 C46 42 28 52 28 52 C28 52 10 42 10 30 V14 Z"
        fill={`url(#${id}-fill)`} stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
      {/* Inner shield line */}
      <path d="M28 10 L40 17 V30 C40 38 28 46 28 46 C28 46 16 38 16 30 V17 Z"
        fill="none" stroke={color} strokeWidth="0.5" opacity="0.15" />
      {/* Scan beam */}
      <rect x="10" y="0" width="36" height="16" fill={`url(#${id}-scan)`} clipPath={`url(#${id}-clip)`}>
        <animate attributeName="y" values="-16;52;-16" dur="3s" repeatCount="indefinite" />
      </rect>
      {/* Exclamation */}
      <line x1="28" y1="20" x2="28" y2="34" stroke={color} strokeWidth="3" strokeLinecap="round">
        <animate attributeName="opacity" values="1;0.35;1" dur="1.8s" repeatCount="indefinite" />
      </line>
      <circle cx="28" cy="40" r="2" fill={color}>
        <animate attributeName="opacity" values="1;0.35;1" dur="1.8s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

/* 3 · Speaker Diarization — Two speakers with alternating waves */
export function DiarizationIcon({ color = '#10b981', size = 56 }: { color?: string; size?: number }) {
  const color2 = '#6366f1';
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Left speaker */}
      <circle cx="16" cy="17" r="6" fill={color} fillOpacity="0.15" stroke={color} strokeWidth="1.3" />
      <circle cx="16" cy="15" r="2.5" fill={color} opacity="0.4" />
      <path d="M7 44 C7 33 11 30 16 30 C21 30 25 33 25 44" fill={color} fillOpacity="0.08" stroke={color} strokeWidth="1.2" strokeLinecap="round" />
      {/* Right speaker */}
      <circle cx="40" cy="17" r="6" fill={color2} fillOpacity="0.15" stroke={color2} strokeWidth="1.3" />
      <circle cx="40" cy="15" r="2.5" fill={color2} opacity="0.4" />
      <path d="M31 44 C31 33 35 30 40 30 C45 30 49 33 49 44" fill={color2} fillOpacity="0.08" stroke={color2} strokeWidth="1.2" strokeLinecap="round" />
      {/* Left sound waves */}
      <path d="M24 13 C26 17 26 21 24 25" stroke={color} strokeWidth="1.3" fill="none" strokeLinecap="round">
        <animate attributeName="opacity" values="0;0.8;0" dur="2.2s" repeatCount="indefinite" />
      </path>
      <path d="M27 10 C30 17 30 21 27 28" stroke={color} strokeWidth="1" fill="none" strokeLinecap="round">
        <animate attributeName="opacity" values="0;0.45;0" dur="2.2s" begin="0.25s" repeatCount="indefinite" />
      </path>
      {/* Right sound waves */}
      <path d="M32 13 C30 17 30 21 32 25" stroke={color2} strokeWidth="1.3" fill="none" strokeLinecap="round">
        <animate attributeName="opacity" values="0;0.8;0" dur="2.2s" begin="1.1s" repeatCount="indefinite" />
      </path>
      <path d="M29 10 C26 17 26 21 29 28" stroke={color2} strokeWidth="1" fill="none" strokeLinecap="round">
        <animate attributeName="opacity" values="0;0.45;0" dur="2.2s" begin="1.35s" repeatCount="indefinite" />
      </path>
      {/* Center divider */}
      <line x1="28" y1="40" x2="28" y2="50" stroke="white" strokeWidth="1" opacity="0.1" strokeDasharray="2 3">
        <animate attributeName="strokeDashoffset" values="0;-5" dur="1.2s" repeatCount="indefinite" />
      </line>
    </svg>
  );
}

/* 4 · Financial NER — Document with entity highlights */
export function NERIcon({ color = '#f59e0b', size = 56 }: { color?: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Document */}
      <rect x="8" y="4" width="36" height="48" rx="4" fill={color} fillOpacity="0.04" stroke={color} strokeWidth="1.2" />
      <path d="M32 4 V14 H42" stroke={color} strokeWidth="1.2" fill="none" />
      <path d="M32 4 L42 14" stroke={color} strokeWidth="1.2" fill="none" />
      {/* Text lines */}
      {[20, 26, 32, 38, 44].map((y, i) => (
        <line key={y} x1="14" y1={y} x2={32 - (i % 3) * 4} y2={y} stroke={color} strokeWidth="1" opacity="0.15" strokeLinecap="round" />
      ))}
      {/* Highlight 1 — orange (amount) */}
      <rect x="14" y="18" width="16" height="5" rx="1.5" fill={color} fillOpacity="0">
        <animate attributeName="fillOpacity" values="0;0.3;0.3;0" dur="5s" repeatCount="indefinite" />
      </rect>
      {/* Highlight 2 — green (entity) */}
      <rect x="20" y="30" width="14" height="5" rx="1.5" fill="#10b981" fillOpacity="0">
        <animate attributeName="fillOpacity" values="0;0;0.3;0.3;0" dur="5s" repeatCount="indefinite" />
      </rect>
      {/* Highlight 3 — indigo (date) */}
      <rect x="14" y="42" width="12" height="5" rx="1.5" fill="#6366f1" fillOpacity="0">
        <animate attributeName="fillOpacity" values="0;0;0;0.3;0" dur="5s" repeatCount="indefinite" />
      </rect>
      {/* Bracket tag */}
      <g opacity="0.85">
        <animate attributeName="opacity" values="0.6;1;0.6" dur="2.5s" repeatCount="indefinite" />
        <rect x="34" y="1" width="18" height="12" rx="3" fill={color} />
        <text x="43" y="9.5" fontSize="6.5" fill="#fff" textAnchor="middle" fontWeight="700" fontFamily="system-ui">NER</text>
      </g>
    </svg>
  );
}

/* 5 · Sentiment Analysis — Live waveform chart */
export function SentimentIcon({ color = '#f59e0b', size = 56 }: { color?: string; size?: number }) {
  const id = `sent-${Math.random().toString(36).slice(2, 6)}`;
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      <defs>
        <linearGradient id={`${id}-area`} x1="28" y1="10" x2="28" y2="48" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* Axes */}
      <line x1="8" y1="48" x2="50" y2="48" stroke={color} strokeWidth="0.7" opacity="0.2" />
      <line x1="8" y1="48" x2="8" y2="8" stroke={color} strokeWidth="0.7" opacity="0.2" />
      {/* Grid */}
      {[18, 28, 38].map(y => (
        <line key={y} x1="8" y1={y} x2="50" y2={y} stroke={color} strokeWidth="0.4" opacity="0.07" />
      ))}
      {/* Area fill */}
      <path fill={`url(#${id}-area)`}>
        <animate attributeName="d"
          values="M8,44 C14,40 18,34 22,30 C26,26 30,18 34,14 C38,20 42,24 46,20 L50,18 L50,48 L8,48 Z;M8,38 C14,34 18,38 22,32 C26,22 30,16 34,22 C38,28 42,20 46,14 L50,12 L50,48 L8,48 Z;M8,44 C14,40 18,34 22,30 C26,26 30,18 34,14 C38,20 42,24 46,20 L50,18 L50,48 L8,48 Z"
          dur="5s" repeatCount="indefinite" />
      </path>
      {/* Line */}
      <path stroke={color} strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round">
        <animate attributeName="d"
          values="M8,44 C14,40 18,34 22,30 C26,26 30,18 34,14 C38,20 42,24 46,20 L50,18;M8,38 C14,34 18,38 22,32 C26,22 30,16 34,22 C38,28 42,20 46,14 L50,12;M8,44 C14,40 18,34 22,30 C26,26 30,18 34,14 C38,20 42,24 46,20 L50,18"
          dur="5s" repeatCount="indefinite" />
      </path>
      {/* Data points */}
      {[
        { cx: 8, base: 44, alt: 38, delay: '0s' },
        { cx: 22, base: 30, alt: 32, delay: '0.4s' },
        { cx: 34, base: 14, alt: 22, delay: '0.8s' },
        { cx: 50, base: 18, alt: 12, delay: '1.2s' },
      ].map((p, i) => (
        <circle key={i} cx={p.cx} r="3" fill={color}>
          <animate attributeName="cy" values={`${p.base};${p.alt};${p.base}`} dur="5s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.5;1;0.5" dur="2.5s" begin={p.delay} repeatCount="indefinite" />
        </circle>
      ))}
      {/* Emoji indicator */}
      <text x="48" y="12" fontSize="8" opacity="0.7">
        <animate attributeName="opacity" values="0.4;0.8;0.4" dur="3s" repeatCount="indefinite" />
        <tspan>📈</tspan>
      </text>
    </svg>
  );
}

/* 6 · ML Export — Data flowing into structured file */
export function ExportIcon({ color = '#a855f7', size = 56 }: { color?: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* File shape */}
      <path d="M14 22 L14 50 L42 50 L42 22 L34 14 H14 Z" fill={color} fillOpacity="0.04" stroke={color} strokeWidth="1.2" />
      <path d="M34 14 V22 H42" stroke={color} strokeWidth="1.2" fill="none" />
      {/* Structured rows filling in */}
      {[28, 33, 38, 43].map((y, i) => (
        <rect key={y} x="18" y={y} rx="1" height="3" fill={color} opacity="0.3">
          <animate attributeName="width" values={`0;${18 - i * 2}`} dur="2.5s" begin={`${i * 0.35}s`} repeatCount="indefinite" />
          <animate attributeName="opacity" values="0;0.35;0.35;0" dur="2.5s" begin={`${i * 0.35}s`} repeatCount="indefinite" />
        </rect>
      ))}
      {/* JSON bracket hints */}
      <text x="36" y="34" fontSize="8" fill={color} opacity="0.2" fontFamily="monospace">{'{'}</text>
      <text x="36" y="46" fontSize="8" fill={color} opacity="0.2" fontFamily="monospace">{'}'}</text>
      {/* Incoming data particles */}
      {[
        { cx: 22, delay: '0s' },
        { cx: 28, delay: '0.6s' },
        { cx: 34, delay: '1.2s' },
      ].map((p, i) => (
        <React.Fragment key={i}>
          <circle cx={p.cx} r="2.5" fill={color}>
            <animate attributeName="cy" values="2;16" dur="1.8s" begin={p.delay} repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.7;0" dur="1.8s" begin={p.delay} repeatCount="indefinite" />
          </circle>
          {/* Trail */}
          <line x1={p.cx} x2={p.cx} stroke={color} strokeWidth="1" strokeLinecap="round">
            <animate attributeName="y1" values="0;12" dur="1.8s" begin={p.delay} repeatCount="indefinite" />
            <animate attributeName="y2" values="4;16" dur="1.8s" begin={p.delay} repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.3;0" dur="1.8s" begin={p.delay} repeatCount="indefinite" />
          </line>
        </React.Fragment>
      ))}
    </svg>
  );
}


// ═══════════════════════════════════════════════
//  BACKBOARD SECTION ICONS
// ═══════════════════════════════════════════════

/* 7 · LLM Routing — Hub-and-spoke with animated packets */
export function LLMRoutingIcon({ color = '#a855f7', size = 56 }: { color?: string; size?: number }) {
  const nodes = [
    { cx: 12, cy: 12 },
    { cx: 44, cy: 12 },
    { cx: 12, cy: 44 },
    { cx: 44, cy: 44 },
  ];
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Connection lines */}
      {nodes.map((n, i) => (
        <line key={i} x1="28" y1="28" x2={n.cx} y2={n.cy} stroke={color} strokeWidth="1" opacity="0.15" />
      ))}
      {/* Outer nodes */}
      {nodes.map((n, i) => (
        <React.Fragment key={i}>
          <circle cx={n.cx} cy={n.cy} r="5" fill={color} fillOpacity="0.08" stroke={color} strokeWidth="1" />
          <circle cx={n.cx} cy={n.cy} r="2" fill={color} opacity="0.5" />
        </React.Fragment>
      ))}
      {/* Central hub */}
      <circle cx="28" cy="28" r="8" fill={color} fillOpacity="0.1" stroke={color} strokeWidth="1.5">
        <animate attributeName="r" values="8;9.5;8" dur="2.5s" repeatCount="indefinite" />
      </circle>
      <circle cx="28" cy="28" r="3.5" fill={color} opacity="0.7">
        <animate attributeName="opacity" values="0.5;0.9;0.5" dur="2.5s" repeatCount="indefinite" />
      </circle>
      {/* Routing packets */}
      {nodes.map((n, i) => (
        <circle key={`p-${i}`} r="2.5" fill={color}>
          <animateMotion dur="1.8s" begin={`${i * 0.45}s`} repeatCount="indefinite"
            path={`M28,28 L${n.cx},${n.cy}`} />
          <animate attributeName="opacity" values="0;0.9;0" dur="1.8s" begin={`${i * 0.45}s`} repeatCount="indefinite" />
        </circle>
      ))}
      {/* Return packets */}
      {nodes.map((n, i) => (
        <circle key={`r-${i}`} r="1.5" fill={color} opacity="0.4">
          <animateMotion dur="1.8s" begin={`${i * 0.45 + 0.9}s`} repeatCount="indefinite"
            path={`M${n.cx},${n.cy} L28,28`} />
          <animate attributeName="opacity" values="0;0.5;0" dur="1.8s" begin={`${i * 0.45 + 0.9}s`} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
}

/* 8 · Memory — Brain outline with synaptic pulses */
export function MemoryIcon({ color = '#a855f7', size = 56 }: { color?: string; size?: number }) {
  const synapses = [
    { cx: 18, cy: 20, d: '0s' },
    { cx: 38, cy: 22, d: '0.6s' },
    { cx: 22, cy: 34, d: '1.2s' },
    { cx: 36, cy: 32, d: '1.8s' },
    { cx: 28, cy: 26, d: '0.9s' },
  ];
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Brain outline */}
      <path d="M28 6 C18 6 8 14 8 26 C8 38 18 50 28 50 C38 50 48 38 48 26 C48 14 38 6 28 6"
        fill={color} fillOpacity="0.04" stroke={color} strokeWidth="1.3" />
      {/* Hemispheres */}
      <path d="M28 12 C26 20 30 28 28 36 C26 42 28 46 28 48" stroke={color} strokeWidth="0.8" opacity="0.2" fill="none" />
      {/* Left folds */}
      <path d="M13 20 Q18 17 23 22" stroke={color} strokeWidth="0.8" opacity="0.2" fill="none" />
      <path d="M11 30 Q17 27 23 32" stroke={color} strokeWidth="0.8" opacity="0.2" fill="none" />
      <path d="M14 40 Q19 37 24 40" stroke={color} strokeWidth="0.8" opacity="0.15" fill="none" />
      {/* Right folds */}
      <path d="M33 22 Q38 17 43 20" stroke={color} strokeWidth="0.8" opacity="0.2" fill="none" />
      <path d="M33 32 Q39 27 45 30" stroke={color} strokeWidth="0.8" opacity="0.2" fill="none" />
      <path d="M32 40 Q37 37 42 40" stroke={color} strokeWidth="0.8" opacity="0.15" fill="none" />
      {/* Synapse connections */}
      <line x1="18" y1="20" x2="28" y2="26" stroke={color} strokeWidth="0.5" opacity="0.1" />
      <line x1="38" y1="22" x2="28" y2="26" stroke={color} strokeWidth="0.5" opacity="0.1" />
      <line x1="22" y1="34" x2="28" y2="26" stroke={color} strokeWidth="0.5" opacity="0.1" />
      <line x1="36" y1="32" x2="28" y2="26" stroke={color} strokeWidth="0.5" opacity="0.1" />
      {/* Synapse pulses */}
      {synapses.map((s, i) => (
        <circle key={i} cx={s.cx} cy={s.cy} fill={color}>
          <animate attributeName="r" values="1;4;1" dur="2.4s" begin={s.d} repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.8;0;0.8" dur="2.4s" begin={s.d} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
}

/* 9 · RAG — Magnifying glass scanning a document */
export function RAGIcon({ color = '#a855f7', size = 56 }: { color?: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Document */}
      <rect x="4" y="4" width="32" height="44" rx="3.5" fill={color} fillOpacity="0.04" stroke={color} strokeWidth="1.2" />
      {/* Text lines */}
      {[14, 20, 26, 32, 38].map((y, i) => (
        <line key={y} x1="10" y1={y} x2={30 - (i % 2) * 6} y2={y} stroke={color} strokeWidth="1" opacity="0.12" strokeLinecap="round" />
      ))}
      {/* Scanning highlight */}
      <rect x="10" rx="1.5" width="18" height="6" fill={color} fillOpacity="0.25">
        <animate attributeName="y" values="12;18;24;30;36;12" dur="4.5s" repeatCount="indefinite" />
      </rect>
      {/* Magnifying glass */}
      <g>
        <animateTransform attributeName="transform" type="translate" values="0,0;3,-3;0,0" dur="4s" repeatCount="indefinite" />
        <circle cx="40" cy="32" r="10" fill="rgba(11,17,32,0.85)" stroke={color} strokeWidth="1.8" />
        {/* Glass glint */}
        <path d="M34 28 Q36 26 38 28" stroke="white" strokeWidth="0.8" opacity="0.2" fill="none" />
        {/* Outer ring pulse */}
        <circle cx="40" cy="32" r="10" fill="none" stroke={color} strokeWidth="1" opacity="0.3">
          <animate attributeName="r" values="10;12;10" dur="2.5s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.3;0;0.3" dur="2.5s" repeatCount="indefinite" />
        </circle>
        {/* Handle */}
        <line x1="47" y1="39" x2="54" y2="46" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
      </g>
    </svg>
  );
}

/* 10 · Embeddings — Vector space with clustering dots */
export function EmbeddingsIcon({ color = '#a855f7', size = 56 }: { color?: string; size?: number }) {
  const c2 = '#10b981';
  const c3 = '#f59e0b';
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Axes */}
      <line x1="6" y1="50" x2="50" y2="50" stroke={color} strokeWidth="0.8" opacity="0.15" />
      <line x1="6" y1="50" x2="6" y2="6" stroke={color} strokeWidth="0.8" opacity="0.15" />
      <path d="M6 6 L4 10 M6 6 L8 10" stroke={color} strokeWidth="0.8" opacity="0.15" />
      <path d="M50 50 L46 52 M50 50 L46 48" stroke={color} strokeWidth="0.8" opacity="0.15" />
      {/* Cluster 1 — purple, top-right */}
      {[
        { cx: 38, cy: 14, r: 3.5, d: '0s' },
        { cx: 44, cy: 20, r: 3, d: '0.2s' },
        { cx: 36, cy: 10, r: 2.5, d: '0.4s' },
        { cx: 42, cy: 14, r: 2, d: '0.6s' },
      ].map((p, i) => (
        <circle key={`c1-${i}`} cx={p.cx} cy={p.cy} r={p.r} fill={color} opacity="0.45">
          <animate attributeName="cx" values={`${p.cx};${p.cx + 2};${p.cx}`} dur="4s" begin={p.d} repeatCount="indefinite" />
          <animate attributeName="cy" values={`${p.cy};${p.cy - 1.5};${p.cy}`} dur="4s" begin={p.d} repeatCount="indefinite" />
        </circle>
      ))}
      {/* Cluster 2 — green, bottom-left */}
      {[
        { cx: 16, cy: 38, r: 3.5, d: '0.5s' },
        { cx: 22, cy: 42, r: 3, d: '0.7s' },
        { cx: 14, cy: 44, r: 2.5, d: '0.9s' },
        { cx: 20, cy: 36, r: 2, d: '1.1s' },
      ].map((p, i) => (
        <circle key={`c2-${i}`} cx={p.cx} cy={p.cy} r={p.r} fill={c2} opacity="0.45">
          <animate attributeName="cx" values={`${p.cx};${p.cx - 1.5};${p.cx}`} dur="4s" begin={p.d} repeatCount="indefinite" />
          <animate attributeName="cy" values={`${p.cy};${p.cy + 2};${p.cy}`} dur="4s" begin={p.d} repeatCount="indefinite" />
        </circle>
      ))}
      {/* Cluster 3 — amber, middle */}
      {[
        { cx: 26, cy: 26, r: 3, d: '0.3s' },
        { cx: 32, cy: 30, r: 2.5, d: '0.5s' },
        { cx: 28, cy: 32, r: 2, d: '0.8s' },
      ].map((p, i) => (
        <circle key={`c3-${i}`} cx={p.cx} cy={p.cy} r={p.r} fill={c3} opacity="0.45">
          <animate attributeName="cx" values={`${p.cx};${p.cx + 1};${p.cx}`} dur="4s" begin={p.d} repeatCount="indefinite" />
          <animate attributeName="cy" values={`${p.cy};${p.cy - 1};${p.cy}`} dur="4s" begin={p.d} repeatCount="indefinite" />
        </circle>
      ))}
      {/* Cluster boundaries (faint) */}
      <circle cx="40" cy="15" r="10" fill="none" stroke={color} strokeWidth="0.5" opacity="0.08" strokeDasharray="3 3" />
      <circle cx="18" cy="40" r="10" fill="none" stroke={c2} strokeWidth="0.5" opacity="0.08" strokeDasharray="3 3" />
      <circle cx="28" cy="29" r="7" fill="none" stroke={c3} strokeWidth="0.5" opacity="0.08" strokeDasharray="3 3" />
    </svg>
  );
}

/* 11 · Audit Trail — Stacked log entries with checkmarks */
export function AuditTrailIcon({ color = '#a855f7', size = 56 }: { color?: string; size?: number }) {
  const rows = [
    { y: 6, time: '12:01', delay: '0s' },
    { y: 19, time: '12:02', delay: '0.6s' },
    { y: 32, time: '12:03', delay: '1.2s' },
  ];
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {rows.map((r, i) => (
        <React.Fragment key={i}>
          {/* Row card */}
          <rect x="8" y={r.y} width="40" height="10" rx="3" fill={color} fillOpacity="0.04" stroke={color} strokeWidth="1" />
          {/* Text line */}
          <line x1="24" y1={r.y + 5} x2={38 - i * 2} y2={r.y + 5} stroke={color} strokeWidth="1" opacity="0.15" strokeLinecap="round" />
          {/* Timestamp */}
          <text x="46" y={r.y + 7} fontSize="4" fill={color} opacity="0.25" textAnchor="end" fontFamily="system-ui">{r.time}</text>
          {/* Checkmark */}
          <path d={`M13 ${r.y + 4} L15 ${r.y + 6.5} L19 ${r.y + 2}`} stroke="#10b981" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round">
            <animate attributeName="opacity" values="0;0;1;1" dur="3.5s" begin={r.delay} repeatCount="indefinite" />
            <animate attributeName="stroke-dashoffset" values="12;0;0;0" dur="3.5s" begin={r.delay} repeatCount="indefinite" />
          </path>
        </React.Fragment>
      ))}
      {/* New entry sliding in from bottom */}
      <rect x="8" y="45" width="40" height="10" rx="3" fill={color} fillOpacity="0.06" stroke={color} strokeWidth="1">
        <animate attributeName="opacity" values="0;0;0;0.6;0" dur="4.5s" repeatCount="indefinite" />
        <animate attributeName="y" values="52;45;45;45;52" dur="4.5s" repeatCount="indefinite" />
      </rect>
      <circle cx="14" cy="50" r="1.5" fill={color} opacity="0">
        <animate attributeName="opacity" values="0;0;0;0.3;0" dur="4.5s" repeatCount="indefinite" />
        <animate attributeName="cy" values="57;50;50;50;57" dur="4.5s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}


// ═══════════════════════════════════════════════
//  INFRASTRUCTURE SECTION ICONS
// ═══════════════════════════════════════════════

/* 12 · GPU Alpha — Chip with processing cores */
export function GPUAlphaIcon({ color = '#f59e0b', size = 56 }: { color?: string; size?: number }) {
  const pins = [20, 25, 30, 36];
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Chip body */}
      <rect x="14" y="14" width="28" height="28" rx="4" fill={color} fillOpacity="0.06" stroke={color} strokeWidth="1.5" />
      {/* Inner die */}
      <rect x="19" y="19" width="18" height="18" rx="2" fill={color} fillOpacity="0.1" stroke={color} strokeWidth="0.5" />
      {/* 2×2 core grid */}
      {[
        { x: 21, y: 21 },
        { x: 29, y: 21 },
        { x: 21, y: 29 },
        { x: 29, y: 29 },
      ].map((c, i) => (
        <rect key={i} x={c.x} y={c.y} width="6" height="6" rx="1" fill={color}>
          <animate attributeName="opacity" values="0.15;0.6;0.15" dur="1.6s" begin={`${i * 0.4}s`} repeatCount="indefinite" />
        </rect>
      ))}
      {/* Pins — all 4 sides */}
      {pins.map(p => (
        <React.Fragment key={`p-${p}`}>
          <line x1={p} y1="6" x2={p} y2="14" stroke={color} strokeWidth="1.2" opacity="0.3" />
          <line x1={p} y1="42" x2={p} y2="50" stroke={color} strokeWidth="1.2" opacity="0.3" />
          <line x1="6" y1={p} x2="14" y2={p} stroke={color} strokeWidth="1.2" opacity="0.3" />
          <line x1="42" y1={p} x2="50" y2={p} stroke={color} strokeWidth="1.2" opacity="0.3" />
        </React.Fragment>
      ))}
      {/* Activity ripple */}
      <circle cx="28" cy="28" r="4" fill={color} opacity="0">
        <animate attributeName="r" values="4;16;4" dur="2.5s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.2;0;0.2" dur="2.5s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

/* 13 · GPU Bravo — Faster chip with more cores */
export function GPUBravoIcon({ color = '#10b981', size = 56 }: { color?: string; size?: number }) {
  const pins = [17, 22, 27, 32, 37];
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Chip body */}
      <rect x="12" y="12" width="32" height="32" rx="4" fill={color} fillOpacity="0.06" stroke={color} strokeWidth="1.5" />
      {/* Inner die */}
      <rect x="16" y="16" width="24" height="24" rx="2" fill={color} fillOpacity="0.08" stroke={color} strokeWidth="0.5" />
      {/* 3×3 core grid */}
      {[18, 24, 30].map((x, xi) =>
        [18, 24, 30].map((y, yi) => (
          <rect key={`${xi}-${yi}`} x={x} y={y} width="5" height="5" rx="0.8" fill={color}>
            <animate attributeName="opacity" values="0.1;0.65;0.1" dur="1.1s" begin={`${(xi * 3 + yi) * 0.12}s`} repeatCount="indefinite" />
          </rect>
        ))
      )}
      {/* Pins */}
      {pins.map(p => (
        <React.Fragment key={`p-${p}`}>
          <line x1={p} y1="4" x2={p} y2="12" stroke={color} strokeWidth="1" opacity="0.25" />
          <line x1={p} y1="44" x2={p} y2="52" stroke={color} strokeWidth="1" opacity="0.25" />
          <line x1="4" y1={p} x2="12" y2={p} stroke={color} strokeWidth="1" opacity="0.25" />
          <line x1="44" y1={p} x2="52" y2={p} stroke={color} strokeWidth="1" opacity="0.25" />
        </React.Fragment>
      ))}
      {/* Speed streaks */}
      {[20, 28, 36].map((y, i) => (
        <line key={y} x1="0" y1={y} x2="7" y2={y} stroke={color} strokeWidth="1.2" strokeLinecap="round">
          <animate attributeName="opacity" values="0;0.6;0" dur="0.9s" begin={`${i * 0.25}s`} repeatCount="indefinite" />
        </line>
      ))}
    </svg>
  );
}

/* 14 · CPU Cluster — Connected servers */
export function CPUClusterIcon({ color = '#6366f1', size = 56 }: { color?: string; size?: number }) {
  const servers = [
    { x: 4, y: 6 },
    { x: 30, y: 6 },
    { x: 4, y: 38 },
    { x: 30, y: 38 },
  ];
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Connections to center */}
      {servers.map((s, i) => (
        <line key={`conn-${i}`} x1={s.x + 11} y1={s.y + 6} x2="28" y2="28" stroke={color} strokeWidth="0.8" opacity="0.15" strokeDasharray="3 2">
          <animate attributeName="strokeDashoffset" values="0;-5" dur="1.2s" begin={`${i * 0.3}s`} repeatCount="indefinite" />
        </line>
      ))}
      {/* Server boxes */}
      {servers.map((s, i) => (
        <React.Fragment key={i}>
          <rect x={s.x} y={s.y} width="22" height="12" rx="2.5" fill={color} fillOpacity="0.06" stroke={color} strokeWidth="1" />
          {/* Status LED */}
          <circle cx={s.x + 5} cy={s.y + 6} r="2" fill={color}>
            <animate attributeName="opacity" values="0.25;0.8;0.25" dur="1.8s" begin={`${i * 0.35}s`} repeatCount="indefinite" />
          </circle>
          {/* Lines inside */}
          <line x1={s.x + 10} y1={s.y + 4.5} x2={s.x + 18} y2={s.y + 4.5} stroke={color} strokeWidth="0.8" opacity="0.15" strokeLinecap="round" />
          <line x1={s.x + 10} y1={s.y + 7.5} x2={s.x + 16} y2={s.y + 7.5} stroke={color} strokeWidth="0.8" opacity="0.15" strokeLinecap="round" />
        </React.Fragment>
      ))}
      {/* Central hub */}
      <circle cx="28" cy="28" r="5" fill={color} fillOpacity="0.12" stroke={color} strokeWidth="1.2">
        <animate attributeName="r" values="5;6.5;5" dur="2.5s" repeatCount="indefinite" />
      </circle>
      <circle cx="28" cy="28" r="2" fill={color} opacity="0.6" />
      {/* Data packets */}
      {servers.map((s, i) => (
        <circle key={`pk-${i}`} r="2" fill={color}>
          <animateMotion dur="1.5s" begin={`${i * 0.4}s`} repeatCount="indefinite"
            path={`M${s.x + 11},${s.y + 6} L28,28`} />
          <animate attributeName="opacity" values="0;0.7;0" dur="1.5s" begin={`${i * 0.4}s`} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
}

/* 15 · Cloud — Cloud with bidirectional data flow */
export function CloudIcon({ color = '#a855f7', size = 56 }: { color?: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 56 56" fill="none">
      {/* Cloud shape */}
      <path d="M14 36 C6 36 4 31 4 26 C4 21 9 17 14 17 C14 11 19 6 26 6 C33 6 38 11 40 15 C44 14 50 17 50 23 C50 29 45 34 40 34 L40 36 Z"
        fill={color} fillOpacity="0.06" stroke={color} strokeWidth="1.3" strokeLinejoin="round" />
      {/* Inner glow */}
      <ellipse cx="26" cy="24" rx="10" ry="6" fill={color} opacity="0">
        <animate attributeName="opacity" values="0;0.08;0" dur="3.5s" repeatCount="indefinite" />
      </ellipse>
      {/* Upload arrow */}
      <g>
        <animate attributeName="opacity" values="0.3;0.85;0.3" dur="2.5s" repeatCount="indefinite" />
        <line x1="22" y1="46" x2="22" y2="34" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
        <path d="M17 38 L22 33 L27 38" stroke={color} strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </g>
      {/* Download arrow */}
      <g>
        <animate attributeName="opacity" values="0.3;0.85;0.3" dur="2.5s" begin="1.25s" repeatCount="indefinite" />
        <line x1="36" y1="34" x2="36" y2="46" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
        <path d="M31 42 L36 47 L41 42" stroke={color} strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </g>
      {/* Upload particles */}
      {[0, 0.7, 1.4].map((d, i) => (
        <circle key={`up-${i}`} cx="22" r={2 - i * 0.3} fill={color}>
          <animate attributeName="cy" values="50;32" dur="2s" begin={`${d}s`} repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.6;0" dur="2s" begin={`${d}s`} repeatCount="indefinite" />
        </circle>
      ))}
      {/* Download particles */}
      {[0.5, 1.2, 1.9].map((d, i) => (
        <circle key={`dn-${i}`} cx="36" r={2 - i * 0.3} fill={color}>
          <animate attributeName="cy" values="32;50" dur="2s" begin={`${d}s`} repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.6;0" dur="2s" begin={`${d}s`} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
}
