'use client';

import Link from 'next/link';
import {
  ComplianceIcon, FraudIcon, DiarizationIcon, NERIcon, SentimentIcon, ExportIcon,
  LLMRoutingIcon, MemoryIcon, RAGIcon, EmbeddingsIcon, AuditTrailIcon,
  GPUAlphaIcon, GPUBravoIcon, CPUClusterIcon, CloudIcon,
} from './AnimatedIcons';

/* ===== FEATURES SECTION ===== */
export function Features() {
  const features = [
    { Icon: ComplianceIcon, title: 'Compliance Checking', desc: 'Automated RBI guideline verification across every call. Flags missing disclosures and consent gaps.', accent: 'var(--s4)', color: '#ef4444' },
    { Icon: FraudIcon, title: 'Fraud Detection', desc: 'Multi-signal anomaly detection for phishing, social engineering, and unauthorized transaction patterns.', accent: 'var(--s4)', color: '#ef4444' },
    { Icon: DiarizationIcon, title: 'Speaker Diarization', desc: 'Precise agent-customer separation with voice biometric matching and overlap handling.', accent: 'var(--s2)', color: '#10b981' },
    { Icon: NERIcon, title: 'Financial NER', desc: 'Custom-trained entity recognition for amounts, account numbers, dates, and banking terminology.', accent: 'var(--s3)', color: '#f59e0b' },
    { Icon: SentimentIcon, title: 'Sentiment Analysis', desc: 'Real-time emotion tracking across the call. Detects frustration, confusion, and satisfaction shifts.', accent: 'var(--s3)', color: '#f59e0b' },
    { Icon: ExportIcon, title: 'ML-Ready Export', desc: 'Structured JSON and CSV output designed for downstream ML training and analytics pipelines.', accent: 'var(--s5)', color: '#a855f7' },
  ];

  return (
    <section className="section" id="features">
      <div className="container">
        <div className="fade-up" style={{ textAlign: 'center' }}>
          <p className="section-label">Capabilities</p>
          <h2 className="section-title gradient-text">Built for Financial Audio Intelligence</h2>
        </div>
        <div className="features-grid">
          {features.map(f => (
            <div className="feature-card fade-up" key={f.title} style={{ '--accent': f.accent } as React.CSSProperties}>
              <div className="feature-icon">
                <f.Icon color={f.color} size={56} />
              </div>
              <h4>{f.title}</h4>
              <p style={{ color: 'var(--text-dim)', fontSize: '0.875rem', fontWeight: 300 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ===== BACKBOARD SECTION ===== */
export function Backboard() {
  const cells = [
    { Icon: LLMRoutingIcon, title: 'LLM Routing', desc: 'Intelligent model selection per task' },
    { Icon: MemoryIcon, title: 'Memory', desc: 'Cross-call context persistence' },
    { Icon: RAGIcon, title: 'RAG', desc: 'Regulatory knowledge retrieval' },
    { Icon: EmbeddingsIcon, title: 'Embeddings', desc: 'Semantic similarity search' },
    { Icon: AuditTrailIcon, title: 'Audit Trail', desc: 'Full LLM decision logging' },
  ];

  return (
    <section className="section backboard-section" id="backboard">
      <div className="container">
        <div className="fade-up" style={{ textAlign: 'center' }}>
          <p className="section-label" style={{ color: 'var(--s5)', borderColor: 'rgba(168,85,247,0.3)' }}>Powered by Backboard.io</p>
          <h2 className="section-title gradient-text">Intelligent LLM Orchestration</h2>
        </div>
        <div className="backboard-grid fade-up">
          {cells.map((cell, i) => (
            <div key={cell.title} style={{ display: 'contents' }}>
              <div className="backboard-cell">
                <div className="backboard-icon">
                  <cell.Icon color="#a855f7" size={48} />
                </div>
                <h4>{cell.title}</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: 300 }}>{cell.desc}</p>
              </div>
              {i < cells.length - 1 && (
                <div className="backboard-connector"><div className="connector-line" /></div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ===== INFRASTRUCTURE SECTION ===== */
export function Infrastructure() {
  const nodes = [
    { Icon: GPUAlphaIcon, title: 'Alpha \u00B7 RTX 3060', desc: 'Whisper inference, diarization, initial NLU processing', accent: 'var(--s3)', color: '#f59e0b', badgeClass: 'badge-warning', badgeText: 'GPU', spec: '12GB VRAM' },
    { Icon: GPUBravoIcon, title: 'Bravo \u00B7 RTX 4050', desc: 'Advanced NER, sentiment models, fraud pattern detection', accent: 'var(--s2)', color: '#10b981', badgeClass: 'badge-success', badgeText: 'GPU', spec: '8GB VRAM' },
    { Icon: CPUClusterIcon, title: 'CPU Cluster', desc: 'Audio preprocessing, rule engine, data export, API serving', accent: 'var(--s1)', color: '#6366f1', badgeClass: 'badge-info', badgeText: 'CPU', spec: '32 cores' },
    { Icon: CloudIcon, title: 'Backboard Cloud', desc: 'LLM orchestration, RAG, embeddings, audit storage', accent: 'var(--s5)', color: '#a855f7', badgeClass: 'badge-purple', badgeText: 'Cloud', spec: 'Managed' },
  ];

  return (
    <section className="section" id="infra">
      <div className="container">
        <div className="fade-up" style={{ textAlign: 'center' }}>
          <p className="section-label">Infrastructure</p>
          <h2 className="section-title gradient-text">Distributed Processing Architecture</h2>
        </div>
        <div className="infra-grid fade-up">
          {nodes.map((node, i) => (
            <div key={node.title} style={{ display: 'contents' }}>
              <div className="infra-node" style={{ '--node-accent': node.accent } as React.CSSProperties}>
                <div className="infra-node-icon">
                  <node.Icon color={node.color} size={48} />
                </div>
                <h4>{node.title}</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: 300 }}>{node.desc}</p>
                <div className="infra-specs">
                  <span className={`badge ${node.badgeClass}`}>{node.badgeText}</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{node.spec}</span>
                </div>
              </div>
              {i < nodes.length - 1 && (
                <div className="infra-connector">
                  <svg viewBox="0 0 80 4"><line x1="0" y1="2" x2="80" y2="2" stroke="var(--border-light)" strokeWidth="2" strokeDasharray="6 4" /></svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ===== CTA + FOOTER ===== */
export function CTAFooter() {
  return (
    <>
      <section className="section cta-section">
        <div className="container" style={{ textAlign: 'center' }}>
          <div className="fade-up">
            <h2 className="section-title gradient-text">Ready to process your first call?</h2>
            <p className="section-desc" style={{ margin: '0 auto var(--sp-8)', fontStyle: 'italic' }}>
              Upload an audio file and watch FinSight transform it into structured, auditable intelligence in real time.
            </p>
            <Link href="/dashboard" className="btn btn-primary btn-lg">
              Launch Dashboard &rarr;
            </Link>
          </div>
        </div>
      </section>
      <footer className="footer">
        <div className="container footer-inner">
          <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem', letterSpacing: '0.1em', textTransform: 'uppercase' as const }}>
            FIN<span style={{ color: 'var(--orange)' }}>SIGHT</span> &copy; 2024
          </span>
          <div className="footer-links">
            <a href="#" style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>GitHub</a>
            <a href="#" style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Docs</a>
            <a href="#" style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Contact</a>
          </div>
        </div>
      </footer>
    </>
  );
}
