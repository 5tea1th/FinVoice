'use client';

import Link from 'next/link';
import TrybankSvg from './TrybankSvg';

export default function Hero() {
  return (
    <section className="hero-redesigned" id="hero">
      <div className="hero-glow" />

      <div className="hero-split container">
        {/* Left — text content */}
        <div className="hero-left">
          <span className="hero-kicker">Audio Intelligence Pipeline</span>

          <div className="hero-title-block">
            <h1 className="hero-title-text">
              FinVoice
            </h1>
            <p className="hero-tagline">
              Raw bank calls &rarr; structured, auditable, ML-trainable data
            </p>
          </div>

          <div className="hero-desc-block">
            <p className="hero-desc-text">
              An end-to-end audio intelligence pipeline that transforms raw financial call recordings
              into structured, compliant, and ML-ready datasets. Built for Indian banking at scale.
            </p>
            <div className="hero-cta-row">
              <Link href="/dashboard" className="btn btn-primary btn-lg">
                Launch Dashboard &rarr;
              </Link>
              <a href="#pipeline" className="btn btn-outline btn-lg">
                See Pipeline
              </a>
            </div>
            <div className="hero-stats-row">
              <div className="hero-stat">
                <span className="hero-stat-val">1,247</span>
                <span className="hero-stat-label">Calls Processed</span>
              </div>
              <div className="hero-stat">
                <span className="hero-stat-val">81.4%</span>
                <span className="hero-stat-label">Avg Compliance</span>
              </div>
              <div className="hero-stat">
                <span className="hero-stat-val">5</span>
                <span className="hero-stat-label">Pipeline Stages</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right — Lottie SVG animation */}
        <div className="hero-right">
          <div className="hero-big-circle">
            <div className="hero-big-circle-glow" />
            <TrybankSvg />
          </div>
        </div>
      </div>
    </section>
  );
}
