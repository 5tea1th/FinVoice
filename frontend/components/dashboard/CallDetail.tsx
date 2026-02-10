'use client';

import { useEffect, useState, useRef } from 'react';
import {
  getCallById, getTranscript, getEntities, getIntents, getCompliance, getFraudSignals, getEmotions,
  getObligations, getTamperSignals, getAudioUrl,
  type Call, type TranscriptMessage, type Entity, type Intent, type ComplianceRule, type FraudSignal, type EmotionData,
  type Obligation, type TamperSignal,
} from '@/lib/api';

interface CallDetailProps {
  callId: string;
  onBack: () => void;
}

type AnalysisTab = 'pipeline' | 'entities' | 'intents' | 'compliance' | 'fraud' | 'emotions' | 'corrections';

// ── Helpers ──

function scoreColor(score: number): string {
  if (score >= 70) return 'var(--success)';
  if (score >= 40) return 'var(--warning)';
  return 'var(--danger)';
}

const EMOTION_COLORS: Record<string, string> = {
  angry: '#ef4444', disgusted: '#a855f7', fearful: '#f59e0b', happy: '#22c55e',
  neutral: '#64748b', sad: '#3b82f6', surprised: '#06b6d4', other: '#94a3b8', '<unk>': '#6b7280',
};

function QualityBar({ label, score, weight }: { label: string; score: number; weight: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)', marginBottom: '6px' }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', width: '90px', textAlign: 'right' }}>
        {label} <span style={{ opacity: 0.5 }}>({weight})</span>
      </span>
      <div style={{ flex: 1, height: '14px', background: 'var(--bg)', borderRadius: '7px', overflow: 'hidden', border: '1px solid var(--border)' }}>
        <div style={{
          width: `${score}%`, height: '100%', background: scoreColor(score),
          borderRadius: '7px', transition: 'width 0.5s ease',
        }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 600, width: '32px', color: scoreColor(score) }}>{score}</span>
    </div>
  );
}

function SentimentBar({ values, label }: { values: number[]; label: string }) {
  if (!values || values.length === 0) return null;
  return (
    <div style={{ marginBottom: 'var(--sp-3)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>
          avg: {(values.reduce((a, b) => a + b, 0) / values.length).toFixed(2)}
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1px', height: '32px', position: 'relative' }}>
        {/* Zero line */}
        <div style={{ position: 'absolute', top: '50%', left: 0, right: 0, height: '1px', background: 'var(--border)', zIndex: 0 }} />
        {values.map((val, i) => {
          const height = Math.abs(val) * 50;
          const isPositive = val >= 0;
          return (
            <div key={i} style={{ flex: 1, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', position: 'relative', zIndex: 1 }}>
              <div style={{
                position: 'absolute',
                left: 0, right: 0,
                [isPositive ? 'bottom' : 'top']: '50%',
                height: `${Math.max(height, 2)}%`,
                background: val > 0.2 ? 'var(--success)' : val < -0.2 ? 'var(--danger)' : 'var(--warning)',
                borderRadius: isPositive ? '2px 2px 0 0' : '0 0 2px 2px',
                opacity: 0.8,
                minHeight: '2px',
              }} />
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: '2px' }}>
        <span>Start</span>
        <span style={{ fontSize: '0.6rem', opacity: 0.6 }}>+1 positive / -1 negative</span>
        <span>End</span>
      </div>
    </div>
  );
}


export default function CallDetail({ callId, onBack }: CallDetailProps) {
  const [call, setCall] = useState<Call | null>(null);
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [intents, setIntents] = useState<Intent[]>([]);
  const [compliance, setCompliance] = useState<ComplianceRule[]>([]);
  const [fraud, setFraud] = useState<FraudSignal[]>([]);
  const [emotions, setEmotions] = useState<EmotionData | null>(null);
  const [obligations, setObligations] = useState<Obligation[]>([]);
  const [tamperSignals, setTamperSignals] = useState<TamperSignal[]>([]);
  const [tamperRisk, setTamperRisk] = useState('none');
  const [activeTab, setActiveTab] = useState<AnalysisTab>('pipeline');
  const [loading, setLoading] = useState(true);
  const [correctionNotes, setCorrectionNotes] = useState('');
  const [correctionSaved, setCorrectionSaved] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    Promise.all([
      getCallById(callId).then(setCall),
      getTranscript(callId).then(setTranscript),
      getEntities(callId).then(setEntities),
      getIntents(callId).then(setIntents),
      getCompliance(callId).then(setCompliance),
      getFraudSignals(callId).then(setFraud),
      getEmotions(callId).then(setEmotions),
      getObligations(callId).then(setObligations),
      getTamperSignals(callId).then(({ signals, risk }) => {
        setTamperSignals(signals);
        setTamperRisk(risk);
      }),
    ]).finally(() => setLoading(false));
  }, [callId]);

  const handleSeekAudio = (startSec: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = startSec;
      audioRef.current.play();
    }
  };

  const handleSubmitCorrections = async () => {
    try {
      const res = await fetch(`/api/calls/${callId}/corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: correctionNotes }),
      });
      if (res.ok) {
        setCorrectionSaved(true);
        setTimeout(() => setCorrectionSaved(false), 3000);
      }
    } catch { /* ignore */ }
  };

  if (loading) {
    return (
      <div>
        <button className="btn btn-ghost" onClick={onBack} style={{ marginBottom: 'var(--sp-4)', fontFamily: 'var(--font-mono)' }}>
          &larr; Back to calls
        </button>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 'var(--sp-12)', color: 'var(--text-dim)' }}>
          <span style={{ fontFamily: 'var(--font-mono)' }}>Loading call analysis...</span>
        </div>
      </div>
    );
  }

  if (!call) {
    return (
      <div>
        <button className="btn btn-ghost" onClick={onBack} style={{ marginBottom: 'var(--sp-4)', fontFamily: 'var(--font-mono)' }}>
          &larr; Back to calls
        </button>
        <div style={{ padding: 'var(--sp-8)', textAlign: 'center', color: 'var(--text-dim)' }}>
          <p style={{ fontFamily: 'var(--font-mono)' }}>Call not found.</p>
        </div>
      </div>
    );
  }

  const compColor = call.complianceScore >= 80 ? 'var(--success)' : call.complianceScore >= 60 ? 'var(--warning)' : 'var(--danger)';
  const circumference = 2 * Math.PI * 34;
  const offset = circumference - (call.complianceScore / 100) * circumference;
  const riskClass = call.fraudRisk === 'high' ? 'badge-danger' : call.fraudRisk === 'medium' ? 'badge-warning' : 'badge-success';
  const qualityColor = scoreColor(call.audioQualityScore);
  const tamperBadge = tamperRisk === 'high' ? 'badge-danger' : tamperRisk === 'medium' ? 'badge-warning' : 'badge-success';

  const wordCount = transcript.reduce((sum, msg) => sum + msg.text.replace(/<[^>]+>/g, '').split(/\s+/).length, 0);
  const silencePct = Math.max(0, 100 - call.agentTalkPct - call.customerTalkPct);

  // Pipeline timings
  const timingEntries = Object.entries(call.pipelineTimings);
  const totalPipelineTime = timingEntries.reduce((sum, [, v]) => sum + v, 0);

  const STAGE_COLORS = ['#3b82f6', '#06b6d4', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6'];

  return (
    <div>
      <button className="btn btn-ghost" onClick={onBack} style={{ marginBottom: 'var(--sp-4)', fontFamily: 'var(--font-mono)' }}>
        &larr; Back to calls
      </button>

      {/* Header */}
      <div className="call-detail-header">
        <div className="compliance-gauge">
          <svg width="80" height="80" className="gauge-circle">
            <circle cx="40" cy="40" r="34" className="gauge-bg" />
            <circle cx="40" cy="40" r="34" className="gauge-fill" stroke={compColor}
              strokeDasharray={circumference} strokeDashoffset={offset} />
          </svg>
          <div className="gauge-text">
            <span className="gauge-value" style={{ fontFamily: 'var(--font-mono)' }}>{call.complianceScore}</span>
            <span className="gauge-label" style={{ fontFamily: 'var(--font-mono)' }}>Score</span>
          </div>
        </div>
        <div className="call-meta">
          <h3 style={{ fontFamily: 'var(--font-mono)' }}>{call.id}</h3>
          <div className="call-meta-row">
            <span className="call-meta-item"><span>Date:</span> {call.date} {call.time}</span>
            <span className="call-meta-item"><span>Duration:</span> {call.duration}</span>
            <span className="call-meta-item"><span>Language:</span> {call.language.toUpperCase()}</span>
            <span className="call-meta-item"><span>Speakers:</span> {call.numSpeakers}</span>
          </div>
        </div>
        <span className={`badge ${riskClass}`} style={{ fontSize: '0.8rem', padding: '4px 12px', fontFamily: 'var(--font-mono)' }}>
          {call.fraudRisk.toUpperCase()} RISK
        </span>
      </div>

      {/* ── Metrics Row ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 'var(--sp-3)', marginBottom: 'var(--sp-4)' }}>
        {/* Audio Quality Card */}
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: 'var(--sp-3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--sp-2)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>Audio Quality</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-1)' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700, color: qualityColor }}>{call.audioQualityScore}</span>
              <span className={`badge ${call.audioQualityScore >= 70 ? 'badge-success' : call.audioQualityScore >= 40 ? 'badge-warning' : 'badge-danger'}`} style={{ fontSize: '0.6rem' }}>
                {call.audioQualityFlag}
              </span>
            </div>
          </div>
          <QualityBar label="SNR" score={call.audioQualityComponents.snr} weight="40%" />
          <QualityBar label="Clipping" score={call.audioQualityComponents.clipping} weight="20%" />
          <QualityBar label="Speech Ratio" score={call.audioQualityComponents.speech_ratio} weight="20%" />
          <QualityBar label="Spectral" score={call.audioQualityComponents.spectral} weight="20%" />
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '4px' }}>
            SNR: {call.snrDb.toFixed(1)} dB | Speech: {call.speechPercentage.toFixed(0)}%
          </div>
        </div>

        {/* Speaker Distribution Card */}
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: 'var(--sp-3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--sp-2)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>Speaker Distribution</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>{transcript.length} segments</span>
          </div>
          {/* Stacked bar */}
          <div style={{ display: 'flex', height: '20px', borderRadius: '10px', overflow: 'hidden', marginBottom: 'var(--sp-2)', border: '1px solid var(--border)' }}>
            <div style={{ width: `${call.agentTalkPct}%`, background: '#3b82f6', transition: 'width 0.5s' }}
              title={`Agent: ${call.agentTalkPct.toFixed(1)}%`} />
            <div style={{ width: `${call.customerTalkPct}%`, background: '#22c55e', transition: 'width 0.5s' }}
              title={`Customer: ${call.customerTalkPct.toFixed(1)}%`} />
            <div style={{ width: `${silencePct}%`, background: 'var(--bg)', transition: 'width 0.5s' }}
              title={`Silence: ${silencePct.toFixed(1)}%`} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
            <span style={{ color: '#3b82f6' }}>Agent {call.agentTalkPct.toFixed(0)}%</span>
            <span style={{ color: '#22c55e' }}>Customer {call.customerTalkPct.toFixed(0)}%</span>
            <span style={{ color: 'var(--text-dim)' }}>Silence {silencePct.toFixed(0)}%</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 'var(--sp-2)' }}>
            {wordCount.toLocaleString()} words | Conf: {(call.overallTranscriptConfidence * 100).toFixed(0)}%
            {call.numLowConfidenceSegments > 0 && (
              <span style={{ color: 'var(--warning)' }}> ({call.numLowConfidenceSegments} low)</span>
            )}
          </div>
        </div>

        {/* Tamper & Security Card */}
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: 'var(--sp-3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--sp-2)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>Security & Integrity</span>
            <span className={`badge ${tamperBadge}`} style={{ fontSize: '0.6rem' }}>{tamperRisk.toUpperCase()}</span>
          </div>
          <div style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
              <span style={{ color: 'var(--text-dim)' }}>Tamper Signals</span>
              <span style={{ color: tamperSignals.length > 0 ? 'var(--warning)' : 'var(--success)' }}>{tamperSignals.length}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
              <span style={{ color: 'var(--text-dim)' }}>Fraud Signals</span>
              <span style={{ color: fraud.length > 0 ? 'var(--warning)' : 'var(--success)' }}>{fraud.length}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
              <span style={{ color: 'var(--text-dim)' }}>PII Detected</span>
              <span>{entities.filter(e => e.type.startsWith('PII_')).length}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-dim)' }}>Compliance Checks</span>
              <span>
                <span style={{ color: 'var(--success)' }}>{compliance.filter(c => c.passed).length}</span>
                /
                <span style={{ color: compliance.some(c => !c.passed) ? 'var(--danger)' : 'var(--text)' }}>{compliance.length}</span>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Pipeline Timing Bar ── */}
      {timingEntries.length > 0 && (
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: 'var(--sp-2) var(--sp-3)', marginBottom: 'var(--sp-4)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>Pipeline Execution</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>{totalPipelineTime.toFixed(0)}s total</span>
          </div>
          <div style={{ display: 'flex', height: '24px', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)' }}>
            {timingEntries.map(([stage, seconds], i) => {
              const pct = (seconds / totalPipelineTime) * 100;
              if (pct < 0.5) return null;
              return (
                <div key={stage} style={{
                  width: `${pct}%`, background: STAGE_COLORS[i % STAGE_COLORS.length],
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  overflow: 'hidden', cursor: 'default',
                }} title={`${stage}: ${seconds}s (${pct.toFixed(0)}%)`}>
                  {pct > 8 && (
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: '#fff', whiteSpace: 'nowrap' }}>
                      {seconds}s
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 12px', marginTop: '6px' }}>
            {timingEntries.map(([stage, seconds], i) => (
              <span key={stage} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: STAGE_COLORS[i % STAGE_COLORS.length], display: 'inline-block' }} />
                {stage.replace('Stage ', 'S').replace(': ', ': ')}: {seconds}s
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Summary & Key Insights */}
      {call.summary && (
        <div className="panel" style={{ marginBottom: 'var(--sp-4)' }}>
          <div className="panel-header">
            <h3 style={{ fontFamily: 'var(--font-mono)' }}>Call Summary</h3>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              {call.numSpeakers} speakers &middot; {call.duration}
            </span>
          </div>
          <div style={{ padding: 'var(--sp-3)' }}>
            <p style={{ fontSize: '0.875rem', lineHeight: 1.6, color: 'var(--text)', marginBottom: 'var(--sp-3)' }}>
              {call.summary}
            </p>
            {call.keyOutcomes.length > 0 && (
              <div style={{ marginBottom: 'var(--sp-3)' }}>
                <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', marginBottom: 'var(--sp-1)', color: 'var(--accent-primary)' }}>Key Outcomes</h4>
                <ul style={{ paddingLeft: 'var(--sp-4)', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.8 }}>
                  {call.keyOutcomes.map((o, i) => <li key={i}>{o}</li>)}
                </ul>
              </div>
            )}
            {call.nextActions.length > 0 && (
              <div>
                <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', marginBottom: 'var(--sp-1)', color: 'var(--warning)' }}>Next Actions</h4>
                <ul style={{ paddingLeft: 'var(--sp-4)', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.8 }}>
                  {call.nextActions.map((a, i) => <li key={i}>{a}</li>)}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="call-detail-content">
        {/* Transcript with Audio Player */}
        <div className="panel">
          <div className="panel-header">
            <h3 style={{ fontFamily: 'var(--font-mono)' }}>Transcript</h3>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              {transcript.length} segments &middot; {wordCount.toLocaleString()} words
            </span>
          </div>

          {/* Turn-taking visualization */}
          {transcript.length > 0 && (
            <div style={{ padding: '0 var(--sp-3)', marginBottom: 'var(--sp-1)' }}>
              <div style={{ display: 'flex', height: '8px', borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--border)' }}>
                {transcript.map((msg, i) => (
                  <div key={i} style={{
                    flex: 1,
                    background: msg.speaker === 'agent' ? '#3b82f6' : msg.speaker === 'customer' ? '#22c55e' : '#94a3b8',
                    opacity: msg.confidence < 0.7 ? 0.4 : 0.8,
                  }} title={`${msg.name} @ ${msg.time} (conf: ${(msg.confidence * 100).toFixed(0)}%)`} />
                ))}
              </div>
            </div>
          )}

          {/* Audio Player */}
          <div style={{ padding: '0 var(--sp-3) var(--sp-2)' }}>
            <audio ref={audioRef} src={getAudioUrl(callId)} controls style={{ width: '100%', height: '36px' }} />
          </div>

          <div className="transcript-content">
            {transcript.length === 0 ? (
              <div style={{ padding: 'var(--sp-6)', textAlign: 'center', color: 'var(--text-dim)' }}>
                <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>No transcript available.</p>
              </div>
            ) : (
              transcript.map((msg, i) => {
                const text = msg.text.replace(
                  /<entity type="([^"]+)">([^<]+)<\/entity>/g,
                  '<span class="entity" title="$1">$2</span>'
                );
                const isLowConf = msg.confidence < 0.7;
                return (
                  <div
                    className="transcript-msg"
                    key={i}
                    style={isLowConf ? { borderLeft: '3px solid var(--warning)', background: 'rgba(255, 165, 0, 0.04)' } : undefined}
                  >
                    <span
                      className="msg-time"
                      style={{ fontFamily: 'var(--font-mono)', cursor: 'pointer' }}
                      onClick={() => handleSeekAudio(msg.startSec)}
                      title="Click to play from here"
                    >
                      {msg.time}
                    </span>
                    <div className="msg-body">
                      <div className={`msg-speaker ${msg.speaker}`} style={{ fontFamily: 'var(--font-mono)' }}>
                        {msg.name}
                        {isLowConf && <span style={{ color: 'var(--warning)', fontSize: '0.65rem', marginLeft: '6px' }}>LOW CONF</span>}
                      </div>
                      <div className="msg-text" dangerouslySetInnerHTML={{ __html: text }} />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Analysis Tabs */}
        <div className="panel">
          <div className="analysis-tabs">
            {(['pipeline', 'entities', 'intents', 'compliance', 'fraud', 'emotions', 'corrections'] as AnalysisTab[]).map(tab => (
              <button
                key={tab}
                className={`analysis-tab${activeTab === tab ? ' active' : ''}`}
                onClick={() => setActiveTab(tab)}
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
          <div className="analysis-content">
            {/* ── PIPELINE TAB ── */}
            {activeTab === 'pipeline' && (
              <div style={{ padding: 'var(--sp-2)' }}>
                <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-3)', fontSize: '0.85rem' }}>Pipeline Stage Details</h4>

                {/* Stage cards */}
                {[
                  {
                    name: 'Stage 1: Audio Normalization',
                    tools: 'FFmpeg',
                    outputs: [`16kHz mono WAV (${call.durationSec.toFixed(0)}s)`],
                  },
                  {
                    name: 'Stage 2: Audio Quality Analysis',
                    tools: 'Silero VAD, openSMILE',
                    outputs: [
                      `Score: ${call.audioQualityScore}/100 (${call.audioQualityFlag})`,
                      `SNR: ${call.snrDb.toFixed(1)} dB`,
                      `Speech: ${call.speechPercentage.toFixed(0)}%`,
                      `Components: SNR=${call.audioQualityComponents.snr}, Clip=${call.audioQualityComponents.clipping}, Speech=${call.audioQualityComponents.speech_ratio}, Spectral=${call.audioQualityComponents.spectral}`,
                    ],
                  },
                  {
                    name: 'Stage 2.5: Audio Cleanup',
                    tools: 'librosa, Silero VAD',
                    outputs: ['Dead air removal', 'Hold music detection', 'Speech stitching'],
                  },
                  {
                    name: 'Stage 3: Transcription & Diarization',
                    tools: 'WhisperX large-v3-turbo, pyannote 3.1',
                    outputs: [
                      `${transcript.length} segments, ${wordCount.toLocaleString()} words`,
                      `${call.numSpeakers} speakers detected`,
                      `Confidence: ${(call.overallTranscriptConfidence * 100).toFixed(0)}% avg`,
                      call.numLowConfidenceSegments > 0 ? `${call.numLowConfidenceSegments} low-confidence segments` : 'All segments high confidence',
                    ],
                  },
                  {
                    name: 'Stage 4A-B: Sentiment & Entity Extraction',
                    tools: 'FinBERT, spaCy, regex, fine-tuned NER',
                    outputs: [
                      `${entities.filter(e => !e.type.startsWith('PII_')).length} financial entities`,
                      `${entities.filter(e => e.type.startsWith('PII_')).length} PII entities`,
                      `${obligations.length} obligations detected`,
                    ],
                  },
                  {
                    name: 'Stage 4C-D: Compliance & Fraud',
                    tools: 'Rule engine, Parselmouth, emotion2vec',
                    outputs: [
                      `${compliance.filter(c => c.passed).length}/${compliance.length} checks passed`,
                      `${fraud.length} fraud signals`,
                      `${tamperSignals.length} tamper signals (${tamperRisk} risk)`,
                    ],
                  },
                  {
                    name: 'Stage 4E-H: Emotion & PII Analysis',
                    tools: 'emotion2vec, Presidio, Detoxify',
                    outputs: [
                      emotions?.segmentEmotions.length ? `${emotions.segmentEmotions.length} emotion segments analyzed` : 'Emotion analysis pending',
                      emotions?.dominantEmotion ? `Dominant: ${emotions.dominantEmotion}` : '',
                    ].filter(Boolean),
                  },
                  {
                    name: 'Stage 4I: LLM Intelligence',
                    tools: 'Qwen3 8B via Ollama',
                    outputs: [
                      `${intents.length} intents classified`,
                      call.summary ? 'Summary generated' : 'Summary pending',
                      `${call.keyOutcomes.length} outcomes, ${call.nextActions.length} actions`,
                    ],
                  },
                ].map((stage, i) => {
                  const timing = timingEntries.find(([k]) => k.toLowerCase().includes(stage.name.split(':')[0].toLowerCase().replace('stage ', '')));
                  return (
                    <div key={i} style={{
                      background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                      padding: 'var(--sp-2) var(--sp-3)', marginBottom: 'var(--sp-2)',
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 600 }}>{stage.name}</span>
                        {timing && (
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--accent-primary)', background: 'rgba(59,130,246,0.1)', padding: '1px 8px', borderRadius: '8px' }}>
                            {timing[1]}s
                          </span>
                        )}
                      </div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', marginBottom: '2px' }}>
                        Tools: {stage.tools}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {stage.outputs.map((o, j) => (
                          <div key={j}>{o}</div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* ── ENTITIES TAB ── */}
            {activeTab === 'entities' && (
              <div>
                <div className="entity-list">
                  {entities.length === 0 ? (
                    <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.875rem', padding: 'var(--sp-4)' }}>No entities extracted.</p>
                  ) : entities.map((e, i) => (
                    <div className="entity-tag" key={i}>
                      <span className="entity-type" style={{ fontFamily: 'var(--font-mono)' }}>{e.type}</span>
                      <span className="entity-value" style={{ fontFamily: 'var(--font-mono)' }}>{e.value}</span>
                      <span className="entity-context">{e.context} &middot; {(e.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
                {obligations.length > 0 && (
                  <div style={{ marginTop: 'var(--sp-4)', borderTop: '1px solid var(--border)', paddingTop: 'var(--sp-3)' }}>
                    <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.85rem' }}>
                      Obligations ({obligations.length})
                    </h4>
                    {obligations.map((o, i) => {
                      const strengthBadge = o.strength === 'binding' ? 'badge-danger' : o.strength === 'promise' ? 'badge-warning' : o.strength === 'conditional' ? 'badge-info' : 'badge-success';
                      return (
                        <div key={i} style={{ padding: 'var(--sp-2)', marginBottom: 'var(--sp-1)', background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 'var(--sp-2)', flexWrap: 'wrap' }}>
                          <span className={`badge ${strengthBadge}`} style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>{o.strength}</span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>[{o.speaker}]</span>
                          <span style={{ fontSize: '0.8rem', flex: 1 }}>&ldquo;{o.text}&rdquo;</span>
                          {o.legallySignificant && <span className="badge badge-danger" style={{ fontSize: '0.6rem' }}>LEGAL</span>}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* ── INTENTS TAB ── */}
            {activeTab === 'intents' && (
              <div className="intent-list">
                {intents.length === 0 ? (
                  <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.875rem', padding: 'var(--sp-4)' }}>No intents classified.</p>
                ) : intents.map((intent, i) => (
                  <div className="intent-item" key={i}>
                    <div className="intent-label" style={{ fontFamily: 'var(--font-mono)' }}>{intent.intent.replace(/_/g, ' ')}</div>
                    <div className="intent-text">&ldquo;{intent.utterance}&rdquo;</div>
                    <div className="intent-confidence" style={{ fontFamily: 'var(--font-mono)' }}>Confidence: {(intent.confidence * 100).toFixed(0)}%</div>
                  </div>
                ))}
              </div>
            )}

            {/* ── COMPLIANCE TAB ── */}
            {activeTab === 'compliance' && (
              <div className="compliance-list">
                {compliance.length === 0 ? (
                  <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.875rem', padding: 'var(--sp-4)' }}>No compliance checks available.</p>
                ) : compliance.map((c, i) => (
                  <div className="compliance-rule" key={i}>
                    <span className={`compliance-icon ${c.passed ? 'pass' : 'fail'}`}>
                      {c.passed ? '\u2713' : '\u2717'}
                    </span>
                    <div>
                      <h4 style={{ fontFamily: 'var(--font-mono)' }}>{c.rule}</h4>
                      <p>{c.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* ── FRAUD TAB ── */}
            {activeTab === 'fraud' && (
              <div>
                <div className="fraud-list">
                  {fraud.length === 0 ? (
                    <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.875rem', padding: 'var(--sp-4)' }}>No fraud signals detected.</p>
                  ) : fraud.map((f, i) => {
                    const badge = f.risk === 'none' ? 'badge-info' : f.risk === 'low' ? 'badge-success' : f.risk === 'medium' ? 'badge-warning' : 'badge-danger';
                    return (
                      <div className="fraud-signal" key={i}>
                        <span className={`fraud-risk-indicator ${f.risk}`} />
                        <div>
                          <h4 style={{ fontSize: '0.85rem', fontFamily: 'var(--font-mono)' }}>{f.signal}</h4>
                          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{f.detail}</p>
                        </div>
                        <span className={`badge ${badge}`} style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)' }}>{f.risk}</span>
                      </div>
                    );
                  })}
                </div>
                {tamperSignals.length > 0 && (
                  <div style={{ marginTop: 'var(--sp-4)', borderTop: '1px solid var(--border)', paddingTop: 'var(--sp-3)' }}>
                    <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.85rem' }}>
                      Tamper Signals ({tamperSignals.length})
                      <span className={`badge ${tamperBadge}`} style={{ fontSize: '0.6rem', marginLeft: 'var(--sp-2)' }}>{tamperRisk.toUpperCase()} RISK</span>
                    </h4>
                    {tamperSignals.map((t, i) => {
                      const sevBadge = t.severity === 'high' ? 'badge-danger' : t.severity === 'medium' ? 'badge-warning' : 'badge-success';
                      return (
                        <div key={i} style={{ padding: 'var(--sp-2)', marginBottom: 'var(--sp-1)', background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
                          <span className={`badge ${sevBadge}`} style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }}>{t.severity}</span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>{t.timestamp.toFixed(1)}s</span>
                          <span style={{ fontSize: '0.8rem', flex: 1 }}>{t.description}</span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>{(t.confidence * 100).toFixed(0)}%</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* ── EMOTIONS TAB (enhanced) ── */}
            {activeTab === 'emotions' && (
              <div className="emotion-analysis">
                {!emotions ? (
                  <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.875rem', padding: 'var(--sp-4)' }}>No emotion data available.</p>
                ) : (
                  <>
                    {/* Per-speaker emotion profiles */}
                    {Object.keys(emotions.speakerBreakdown).length > 0 && (
                      <div style={{ marginBottom: 'var(--sp-4)' }}>
                        <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.85rem' }}>Speaker Emotion Profiles</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--sp-3)' }}>
                          {Object.entries(emotions.speakerBreakdown).map(([speaker, profile]) => (
                            <div key={speaker} style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: 'var(--sp-3)' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--sp-2)' }}>
                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 600, textTransform: 'capitalize' }}>{speaker}</span>
                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>{profile.total_segments} seg</span>
                              </div>
                              <div style={{ marginBottom: 'var(--sp-2)' }}>
                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>Dominant: </span>
                                <span style={{
                                  fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 600,
                                  color: EMOTION_COLORS[profile.dominant] || 'var(--text)',
                                }}>{profile.dominant}</span>
                              </div>
                              {/* Mini emotion bars */}
                              {Object.entries(profile.distribution)
                                .sort(([, a], [, b]) => b - a)
                                .map(([emotion, pct]) => (
                                  <div key={emotion} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '3px' }}>
                                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-dim)', width: '55px', textAlign: 'right' }}>{emotion}</span>
                                    <div style={{ flex: 1, height: '10px', background: 'var(--bg)', borderRadius: '5px', overflow: 'hidden' }}>
                                      <div style={{ width: `${pct * 100}%`, height: '100%', background: EMOTION_COLORS[emotion] || '#94a3b8', borderRadius: '5px' }} />
                                    </div>
                                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-dim)', width: '30px' }}>{(pct * 100).toFixed(0)}%</span>
                                  </div>
                                ))}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Emotion Distribution (call-wide) */}
                    <div style={{ marginBottom: 'var(--sp-4)' }}>
                      <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.85rem' }}>
                        Call-Wide Emotion Distribution
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', marginLeft: 'var(--sp-2)' }}>
                          Dominant: {emotions.dominantEmotion}
                        </span>
                      </h4>
                      {/* Horizontal stacked bar */}
                      {Object.keys(emotions.emotionDistribution).length > 0 && (
                        <div style={{ marginBottom: 'var(--sp-2)' }}>
                          <div style={{ display: 'flex', height: '24px', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)' }}>
                            {Object.entries(emotions.emotionDistribution)
                              .sort(([, a], [, b]) => b - a)
                              .map(([emotion, pct]) => (
                                <div key={emotion} style={{
                                  width: `${pct * 100}%`, background: EMOTION_COLORS[emotion] || '#94a3b8',
                                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                                }} title={`${emotion}: ${(pct * 100).toFixed(0)}%`}>
                                  {pct > 0.08 && (
                                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: '#fff' }}>{emotion}</span>
                                  )}
                                </div>
                              ))}
                          </div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 12px', marginTop: '6px' }}>
                            {Object.entries(emotions.emotionDistribution)
                              .sort(([, a], [, b]) => b - a)
                              .map(([emotion, pct]) => (
                                <span key={emotion} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: EMOTION_COLORS[emotion] || '#94a3b8' }} />
                                  {emotion}: {(pct * 100).toFixed(0)}%
                                </span>
                              ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Segment-by-segment emotion timeline */}
                    {emotions.segmentEmotions.length > 0 && (
                      <div style={{ marginBottom: 'var(--sp-4)' }}>
                        <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.85rem' }}>
                          Emotion Timeline ({emotions.segmentEmotions.length} segments)
                        </h4>
                        <div style={{ display: 'flex', gap: '2px', flexWrap: 'wrap' }}>
                          {emotions.segmentEmotions.map((seg, i) => (
                            <div key={i} style={{
                              width: '14px', height: '14px', borderRadius: '3px',
                              background: EMOTION_COLORS[seg.emotion] || '#94a3b8',
                              opacity: Math.max(0.4, seg.score),
                              cursor: 'default',
                            }} title={`#${seg.segment_id} ${seg.speaker}: ${seg.emotion} (${(seg.score * 100).toFixed(0)}%)`} />
                          ))}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                          <span>Start</span>
                          <span>Each square = 1 segment (hover for details)</span>
                          <span>End</span>
                        </div>
                      </div>
                    )}

                    {/* Dual Sentiment Trajectory */}
                    <div>
                      <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-3)', fontSize: '0.85rem' }}>Sentiment Trajectories</h4>
                      <SentimentBar values={emotions.customerTrajectory} label="Customer Sentiment" />
                      <SentimentBar values={emotions.agentTrajectory} label="Agent Sentiment" />
                    </div>

                    {/* Escalation moments */}
                    {emotions.escalationMoments.length > 0 && (
                      <div style={{ marginTop: 'var(--sp-3)', borderTop: '1px solid var(--border)', paddingTop: 'var(--sp-2)' }}>
                        <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.85rem', color: 'var(--danger)' }}>
                          Escalation Moments ({emotions.escalationMoments.length})
                        </h4>
                        {emotions.escalationMoments.map((m, i) => (
                          <div key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', padding: '4px 0', color: 'var(--text-muted)' }}>
                            Segment #{m.segment_id} ({m.speaker}): {m.emotion} at {(m.score * 100).toFixed(0)}% confidence
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {/* ── CORRECTIONS TAB ── */}
            {activeTab === 'corrections' && (
              <div style={{ padding: 'var(--sp-3)' }}>
                <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-3)' }}>Transcript Corrections & Notes</h4>
                <textarea
                  value={correctionNotes}
                  onChange={(e) => setCorrectionNotes(e.target.value)}
                  placeholder="Add corrections, notes, or entity overrides..."
                  style={{
                    width: '100%',
                    minHeight: '120px',
                    padding: 'var(--sp-3)',
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.85rem',
                    color: 'var(--text)',
                    resize: 'vertical',
                  }}
                />
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-3)', marginTop: 'var(--sp-3)' }}>
                  <button
                    className="btn"
                    onClick={handleSubmitCorrections}
                    style={{ fontFamily: 'var(--font-mono)', background: 'var(--accent)', color: 'var(--bg)', padding: 'var(--sp-2) var(--sp-4)', border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer' }}
                  >
                    Save Corrections
                  </button>
                  {correctionSaved && (
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--success)' }}>Saved!</span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
