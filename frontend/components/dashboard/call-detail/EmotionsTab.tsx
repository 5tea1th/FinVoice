'use client';

import { SentimentTrajectoryChart, EmotionRadar, EmotionTimeline } from '@/components/charts';
import type { EmotionData } from '@/lib/api';

const EMOTION_COLORS: Record<string, string> = {
  angry: '#ef4444', disgusted: '#a855f7', fearful: '#f59e0b', happy: '#22c55e',
  neutral: '#64748b', sad: '#3b82f6', surprised: '#06b6d4', other: '#94a3b8', '<unk>': '#6b7280',
};

interface EmotionsTabProps {
  emotions: EmotionData | null;
}

export default function EmotionsTab({ emotions }: EmotionsTabProps) {
  if (!emotions) {
    return (
      <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', padding: 'var(--sp-6)', textAlign: 'center' }}>
        No emotion data available for this call.
      </p>
    );
  }

  return (
    <div className="emotion-analysis">
      {/* Emotion Radar */}
      {Object.keys(emotions.speakerBreakdown).length > 0 && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.78rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Speaker Emotion Radar
          </h4>
          <EmotionRadar speakerBreakdown={emotions.speakerBreakdown} />
        </div>
      )}

      {/* Sentiment Trajectory Chart */}
      {(emotions.customerTrajectory.length > 0 || emotions.agentTrajectory.length > 0) && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.78rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Sentiment Trajectory
          </h4>
          <SentimentTrajectoryChart
            customerTrajectory={emotions.customerTrajectory}
            agentTrajectory={emotions.agentTrajectory}
          />
          <div style={{ display: 'flex', gap: 'var(--sp-4)', justifyContent: 'center', marginTop: 'var(--sp-1)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: '#22c55e', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 12, height: 3, background: '#22c55e', display: 'inline-block', borderRadius: 2 }} /> Customer
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: '#6366f1', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 12, height: 3, background: '#6366f1', display: 'inline-block', borderRadius: 2 }} /> Agent
            </span>
          </div>
        </div>
      )}

      {/* Emotion Flow Timeline */}
      {emotions.segmentEmotions.length > 0 && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.78rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Emotion Flow ({emotions.segmentEmotions.length} segments)
          </h4>
          <EmotionTimeline segmentEmotions={emotions.segmentEmotions} />
        </div>
      )}

      {/* Emotion Distribution Bar */}
      {Object.keys(emotions.emotionDistribution).length > 0 && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.78rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Call-Wide Distribution — Dominant: {emotions.dominantEmotion}
          </h4>
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

      {/* Escalation moments */}
      {emotions.escalationMoments.length > 0 && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 'var(--sp-2)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.78rem', color: 'var(--danger)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Escalation Moments ({emotions.escalationMoments.length})
          </h4>
          {emotions.escalationMoments.map((m, i) => (
            <div key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', padding: '4px 0', color: 'var(--text-muted)' }}>
              Segment #{m.segment_id} ({m.speaker}): {m.emotion} at {(m.score * 100).toFixed(0)}% confidence
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
