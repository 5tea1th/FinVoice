'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { PIPELINE_STAGES } from '@/lib/api';
import StageIllustration from './StageIllustrations';

const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#a855f7'];
const PARTICLE_COUNT = 4;

function NodeIcon({ stage }: { stage: number }) {
  switch (stage) {
    case 0: return (
      <g stroke="#fff" strokeWidth="2" fill="none" strokeLinecap="round">
        <path d="M-12,0 Q-8,-10 -4,0 T4,0 T12,0" />
      </g>
    );
    case 1: return (
      <g stroke="#fff" strokeLinecap="round">
        <line x1="-11" y1="-5" x2="11" y2="-5" strokeWidth="2" />
        <line x1="-11" y1="1" x2="7" y2="1" strokeWidth="2" opacity="0.6" />
        <line x1="-11" y1="7" x2="9" y2="7" strokeWidth="1.5" opacity="0.35" />
      </g>
    );
    case 2: return (
      <g stroke="#fff" fill="none" strokeLinejoin="round">
        <path d="M-7,-9 L7,-9 L10,0 L7,9 L-7,9 Z" strokeWidth="1.8" />
        <circle cx="-1" cy="-3" r="2.5" fill="#fff" opacity="0.5" />
        <circle cx="3" cy="4" r="1.8" fill="#fff" opacity="0.35" />
      </g>
    );
    case 3: return (
      <g stroke="#fff" fill="none" strokeLinejoin="round">
        <path d="M0,-12 L11,-6 V5 C11,11 0,15 0,15 C0,15 -11,11 -11,5 V-6 Z" strokeWidth="1.8" />
        <path d="M-4,1 L-1,4 L6,-3" strokeWidth="2.2" strokeLinecap="round" />
      </g>
    );
    case 4: return (
      <g stroke="#fff" fill="none">
        <ellipse cx="0" cy="-7" rx="11" ry="5" strokeWidth="1.8" />
        <path d="M-11,-7 V5 C-11,11 11,11 11,5 V-7" strokeWidth="1.8" />
        <ellipse cx="0" cy="5" rx="11" ry="5" strokeWidth="0.8" opacity="0.3" />
      </g>
    );
    default: return null;
  }
}

function buildPath(fromX: number, toX: number) {
  const r = 3;
  const mid = 50;
  const goingRight = toX > fromX;
  const rSign = goingRight ? r : -r;
  return `M${fromX},0 V${mid - r} Q${fromX},${mid} ${fromX + rSign},${mid} H${toX - rSign} Q${toX},${mid} ${toX},${mid + r} V100`;
}

export default function Pipeline() {
  const containerRef = useRef<HTMLDivElement>(null);
  const nodeRefs = useRef<(SVGSVGElement | null)[]>([]);
  const [xPos, setXPos] = useState<number[]>([]);

  const measure = useCallback(() => {
    const ctr = containerRef.current;
    if (!ctr) return;
    const cRect = ctr.getBoundingClientRect();
    if (cRect.width === 0) return;
    const positions = nodeRefs.current.map(el => {
      if (!el) return 50;
      const r = el.getBoundingClientRect();
      return ((r.left + r.width / 2 - cRect.left) / cRect.width) * 100;
    });
    setXPos(positions);
  }, []);

  useEffect(() => {
    // Measure after first paint
    const raf = requestAnimationFrame(() => measure());
    const ro = new ResizeObserver(() => measure());
    if (containerRef.current) ro.observe(containerRef.current);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, [measure]);

  return (
    <section className="section" id="pipeline">
      <div className="container">
        <div className="fade-up" style={{ textAlign: 'center' }}>
          <p className="section-label">The Pipeline</p>
          <h2 className="section-title gradient-text">Five Stages. One Stream.</h2>
          <p className="section-desc" style={{ margin: '0 auto var(--sp-12)' }}>
            Every call flows through five specialized processing stages, transforming raw audio into structured, auditable intelligence.
          </p>
        </div>

        <div className="ps-serpentine" ref={containerRef}>
          {PIPELINE_STAGES.map((stage, i) => {
            const isLeft = i % 2 === 0;
            const color = COLORS[i];

            // Measured positions or fallback estimates
            const myX = xPos[i] ?? (isLeft ? 13 : 87);
            const nextX = xPos[i + 1] ?? (isLeft ? 87 : 13);

            const node = (
              <div className="ps-node-wrap">
                <svg
                  viewBox="0 0 120 120"
                  className="ps-node-svg"
                  ref={el => { nodeRefs.current[i] = el; }}
                >
                  <defs>
                    <radialGradient id={`ng-${i}`} cx="40%" cy="35%">
                      <stop offset="0%" stopColor={color} stopOpacity="0.22" />
                      <stop offset="100%" stopColor={color} stopOpacity="0.04" />
                    </radialGradient>
                    <filter id={`glow-${i}`} x="-50%" y="-50%" width="200%" height="200%">
                      <feGaussianBlur in="SourceGraphic" stdDeviation="8" />
                    </filter>
                  </defs>
                  <circle cx="60" cy="60" r="55" fill={color} filter={`url(#glow-${i})`} opacity="0.15">
                    <animate attributeName="opacity" values="0.1;0.2;0.1" dur={`${3 + i * 0.4}s`} repeatCount="indefinite" />
                  </circle>
                  <circle cx="60" cy="60" r="44" fill="none" stroke={color} strokeWidth="1">
                    <animate attributeName="r" values="44;56;44" dur={`${3 + i * 0.3}s`} repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.2;0;0.2" dur={`${3 + i * 0.3}s`} repeatCount="indefinite" />
                  </circle>
                  <circle cx="60" cy="60" r="44" fill={`url(#ng-${i})`} stroke={color} strokeWidth="2" />
                  <circle cx="60" cy="60" r="36" fill="none" stroke={color} strokeWidth="0.5" opacity="0.12" />
                  <path d="M30,38 A44,44 0 0,1 90,38" fill="none" stroke="white" strokeWidth="1" opacity="0.07" />
                  <g transform="translate(60,60)" opacity="0.9">
                    <NodeIcon stage={i} />
                  </g>
                </svg>
                <div className="ps-node-label" style={{ color }}>
                  <span className="ps-node-num">{String(i + 1).padStart(2, '0')}</span>
                  <span className="ps-node-name">{stage.name}</span>
                </div>
              </div>
            );

            const card = (
              <div className="ps-card" style={{ '--stage-color': color } as React.CSSProperties}>
                <h3 className="ps-card-title">{stage.name}</h3>
                <p className="ps-card-desc">{stage.desc}</p>
                <div className="ps-card-tools">
                  {stage.tools.map(t => <span className="tool-tag" key={t}>{t}</span>)}
                </div>
                <div className="pipeline-card-outputs">
                  {stage.outputs.map(o => <span key={o}>{o}</span>)}
                </div>
              </div>
            );

            const illustration = (
              <div className="ps-illustration-wrap">
                <StageIllustration stage={i} color={color} />
              </div>
            );

            return (
              <React.Fragment key={stage.id}>
                <div className={`ps-row ${isLeft ? 'ps-row-left' : 'ps-row-right'} fade-up`}>
                  {isLeft ? <>{node}{card}{illustration}</> : <>{illustration}{card}{node}</>}
                </div>

                {i < PIPELINE_STAGES.length - 1 && (() => {
                  const path = buildPath(myX, nextX);
                  const fromXPct = `${myX}%`;
                  const toXPct = `${nextX}%`;

                  return (
                    <div className="ps-connector-wrap">
                      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="ps-connector-svg">
                        <defs>
                          <linearGradient id={`sc-${i}`} x1={isLeft ? '0' : '1'} y1="0" x2={isLeft ? '1' : '0'} y2="0">
                            <stop offset="0%" stopColor={COLORS[i]} />
                            <stop offset="100%" stopColor={COLORS[i + 1]} />
                          </linearGradient>
                        </defs>
                        <path d={path} stroke={`url(#sc-${i})`} strokeWidth="1" fill="none" opacity="0.12" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
                        <path d={path} stroke={`url(#sc-${i})`} strokeWidth="2" fill="none" opacity="0.35" strokeLinecap="round" strokeDasharray="10 16" vectorEffect="non-scaling-stroke">
                          <animate attributeName="stroke-dashoffset" values="0;-52" dur="2s" repeatCount="indefinite" />
                        </path>
                      </svg>

                      {/* CSS-animated particles — outside SVG so they stay round */}
                      {Array.from({ length: PARTICLE_COUNT }, (_, j) => (
                        <div
                          key={j}
                          className="ps-particle"
                          style={{
                            '--p-color': COLORS[i + (j % 2 === 0 ? 0 : 1)],
                            '--p-delay': `${-(j * (6 / PARTICLE_COUNT)).toFixed(2)}s`,
                            '--from-x': fromXPct,
                            '--to-x': toXPct,
                            '--q1-x': `${myX + (nextX - myX) * 0.25}%`,
                            '--mid-x': `${(myX + nextX) / 2}%`,
                            '--q3-x': `${myX + (nextX - myX) * 0.75}%`,
                            '--p-size': `${5 + (j % 3) * 2}px`,
                          } as React.CSSProperties}
                        />
                      ))}
                    </div>
                  );
                })()}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </section>
  );
}
