'use client';

import { useEffect, useState, useRef } from 'react';
import {
  getCallById, getTranscript, getEntities, getIntents, getCompliance, getFraudSignals, getEmotions,
  getObligations, getTamperSignals, getAudioUrl, downloadCallJson, downloadMaskedTranscript,
  type Call, type TranscriptMessage, type Entity, type Intent, type ComplianceRule, type FraudSignal, type EmotionData,
  type Obligation, type TamperSignal,
} from '@/lib/api';
import InsightsTab from './call-detail/InsightsTab';
import PipelineTab from './call-detail/PipelineTab';
import EntitiesTab from './call-detail/EntitiesTab';
import IntentsTab from './call-detail/IntentsTab';
import ComplianceTab from './call-detail/ComplianceTab';
import FraudTab from './call-detail/FraudTab';
import EmotionsTab from './call-detail/EmotionsTab';
import CorrectionsTab from './call-detail/CorrectionsTab';

interface CallDetailProps {
  callId: string;
  onBack: () => void;
}

type AnalysisTab = 'insights' | 'pipeline' | 'entities' | 'intents' | 'compliance' | 'fraud' | 'emotions' | 'corrections';


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
  const [activeTab, setActiveTab] = useState<AnalysisTab>('insights');
  const [loading, setLoading] = useState(true);
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

  const handleSegmentClick = (segmentId: number) => {
    const el = document.getElementById(`transcript-seg-${segmentId}`);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.style.transition = 'background 0.2s';
    el.style.background = 'rgba(59, 130, 246, 0.15)';
    setTimeout(() => { el.style.background = ''; }, 1500);
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

  const wordCount = transcript.reduce((sum, msg) => sum + msg.text.replace(/<[^>]+>/g, '').split(/\s+/).length, 0);

  return (
    <div>
      {/* Top bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--sp-4)' }}>
        <button className="btn btn-ghost" onClick={onBack} style={{ fontFamily: 'var(--font-mono)' }}>
          &larr; Back to calls
        </button>
        <div style={{ display: 'flex', gap: 'var(--sp-2)' }}>
          <button className="btn btn-ghost" onClick={() => downloadCallJson(callId)} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
            &#x2913; JSON
          </button>
          <button className="btn btn-ghost" onClick={() => downloadMaskedTranscript(callId)} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
            &#x2913; Masked
          </button>
        </div>
      </div>

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

      {/* Compact stats strip */}
      <div className="call-stats-strip">
        {[
          { label: 'Quality', value: `${call.audioQualityScore}`, color: call.audioQualityScore >= 70 ? 'var(--success)' : call.audioQualityScore >= 40 ? 'var(--warning)' : 'var(--danger)' },
          { label: 'Words', value: wordCount.toLocaleString() },
          { label: 'Confidence', value: `${(call.overallTranscriptConfidence * 100).toFixed(0)}%` },
          { label: 'Compliance', value: `${compliance.filter(c => c.passed).length}/${compliance.length}` },
          { label: 'Entities', value: `${entities.length}` },
          { label: 'Intents', value: `${intents.length}` },
          { label: 'Fraud', value: `${fraud.length}`, color: fraud.length > 0 ? 'var(--warning)' : undefined },
          { label: 'Tamper', value: tamperRisk, color: tamperRisk === 'high' ? 'var(--danger)' : tamperRisk === 'medium' ? 'var(--warning)' : undefined },
        ].map((stat, i) => (
          <div key={i} className="call-stat-chip">
            <span className="call-stat-label">{stat.label}</span>
            <span className="call-stat-value" style={stat.color ? { color: stat.color } : undefined}>{stat.value}</span>
          </div>
        ))}
      </div>

      {/* Summary */}
      {call.summary && (
        <div className="panel" style={{ marginBottom: 'var(--sp-4)' }}>
          <div className="panel-header">
            <h3 style={{ fontFamily: 'var(--font-mono)' }}>Summary</h3>
          </div>
          <div style={{ padding: 'var(--sp-3)' }}>
            <p style={{ fontSize: '0.875rem', lineHeight: 1.6, color: 'var(--text)', marginBottom: call.keyOutcomes.length > 0 || call.nextActions.length > 0 ? 'var(--sp-3)' : 0 }}>
              {call.summary}
            </p>
            {(call.keyOutcomes.length > 0 || call.nextActions.length > 0) && (
              <div style={{ display: 'flex', gap: 'var(--sp-5)', flexWrap: 'wrap' }}>
                {call.keyOutcomes.length > 0 && (
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', marginBottom: 'var(--sp-1)', color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Outcomes</h4>
                    <ul style={{ paddingLeft: 'var(--sp-4)', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.8 }}>
                      {call.keyOutcomes.map((o, i) => <li key={i}>{o}</li>)}
                    </ul>
                  </div>
                )}
                {call.nextActions.length > 0 && (
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', marginBottom: 'var(--sp-1)', color: 'var(--warning)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Next Actions</h4>
                    <ul style={{ paddingLeft: 'var(--sp-4)', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.8 }}>
                      {call.nextActions.map((a, i) => <li key={i}>{a}</li>)}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Content: Transcript + Analysis Tabs */}
      <div className="call-detail-content">
        {/* Transcript */}
        <div className="panel">
          <div className="panel-header">
            <h3 style={{ fontFamily: 'var(--font-mono)' }}>Transcript</h3>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              {transcript.length} segments
            </span>
          </div>
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
          <div style={{ padding: '0 var(--sp-3) var(--sp-2)' }}>
            <audio ref={audioRef} src={getAudioUrl(callId)} controls style={{ width: '100%', height: '36px' }} />
          </div>
          <div className="transcript-content">
            {transcript.length === 0 ? (
              <div style={{ padding: 'var(--sp-6)', textAlign: 'center', color: 'var(--text-dim)' }}>
                <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>No transcript available.</p>
              </div>
            ) : transcript.map((msg, i) => {
              const text = msg.text.replace(
                /<entity type="([^"]+)">([^<]+)<\/entity>/g,
                '<span class="entity" title="$1">$2</span>'
              );
              const isLowConf = msg.confidence < 0.7;
              const speakerClass = msg.speaker === 'agent' ? 'speaker-agent' : msg.speaker === 'customer' ? 'speaker-customer' : '';
              return (
                <div className={`transcript-msg ${speakerClass}`} key={i} id={`transcript-seg-${i}`}
                  style={isLowConf ? { borderLeftColor: 'var(--warning)', background: 'rgba(255, 165, 0, 0.04)' } : undefined}
                >
                  <span className="msg-time" style={{ fontFamily: 'var(--font-mono)', cursor: 'pointer' }}
                    onClick={() => handleSeekAudio(msg.startSec)} title="Click to play from here">
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
            })}
          </div>
        </div>

        {/* Analysis Tabs */}
        <div className="panel">
          <div className="analysis-tabs">
            {([
              { key: 'insights', label: 'Insights' },
              { key: 'pipeline', label: 'Pipeline' },
              { key: 'entities', label: 'Entities', count: entities.length },
              { key: 'intents', label: 'Intents', count: intents.length },
              { key: 'compliance', label: 'Compliance', count: compliance.length },
              { key: 'fraud', label: 'Fraud', count: fraud.length },
              { key: 'emotions', label: 'Emotions' },
              { key: 'corrections', label: 'Corrections' },
            ] as { key: AnalysisTab; label: string; count?: number }[]).map(tab => (
              <button
                key={tab.key}
                className={`analysis-tab${activeTab === tab.key ? ' active' : ''}`}
                onClick={() => setActiveTab(tab.key)}
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                {tab.label}
                {tab.count !== undefined && tab.count > 0 && (
                  <span style={{
                    marginLeft: '6px', fontSize: '0.6rem', padding: '1px 5px',
                    borderRadius: '8px', background: 'rgba(255,255,255,0.1)', fontWeight: 600,
                  }}>{tab.count}</span>
                )}
              </button>
            ))}
          </div>
          <div className="analysis-content">
            {activeTab === 'insights' && <InsightsTab financialInsights={call.financialInsights} onSegmentClick={handleSegmentClick} />}
            {activeTab === 'pipeline' && (
              <PipelineTab
                pipelineTimings={call.pipelineTimings}
                durationSec={call.durationSec}
                audioQualityScore={call.audioQualityScore}
                audioQualityFlag={call.audioQualityFlag}
                snrDb={call.snrDb}
                speechPercentage={call.speechPercentage}
                audioQualityComponents={call.audioQualityComponents}
                transcriptLength={transcript.length}
                wordCount={wordCount}
                numSpeakers={call.numSpeakers}
                overallTranscriptConfidence={call.overallTranscriptConfidence}
                numLowConfidenceSegments={call.numLowConfidenceSegments}
                entityCount={entities.filter(e => !e.type.startsWith('PII_')).length}
                piiCount={entities.filter(e => e.type.startsWith('PII_')).length}
                obligationCount={obligations.length}
                compliancePassed={compliance.filter(c => c.passed).length}
                complianceTotal={compliance.length}
                fraudCount={fraud.length}
                tamperCount={tamperSignals.length}
                tamperRisk={tamperRisk}
                emotionSegments={emotions?.segmentEmotions.length || 0}
                dominantEmotion={emotions?.dominantEmotion || ''}
                intentCount={intents.length}
                hasSummary={!!call.summary}
                outcomeCount={call.keyOutcomes.length}
                actionCount={call.nextActions.length}
              />
            )}
            {activeTab === 'entities' && <EntitiesTab entities={entities} obligations={obligations} />}
            {activeTab === 'intents' && <IntentsTab intents={intents} />}
            {activeTab === 'compliance' && <ComplianceTab compliance={compliance} />}
            {activeTab === 'fraud' && <FraudTab fraud={fraud} tamperSignals={tamperSignals} tamperRisk={tamperRisk} />}
            {activeTab === 'emotions' && <EmotionsTab emotions={emotions} />}
            {activeTab === 'corrections' && <CorrectionsTab callId={callId} />}
          </div>
        </div>
      </div>
    </div>
  );
}
