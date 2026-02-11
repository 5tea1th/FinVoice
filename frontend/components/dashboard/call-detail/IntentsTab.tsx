'use client';

import { IntentDistribution, getIntentColor } from '@/components/charts';
import type { Intent } from '@/lib/api';

interface IntentsTabProps {
  intents: Intent[];
}

export default function IntentsTab({ intents }: IntentsTabProps) {
  if (intents.length === 0) {
    return (
      <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', padding: 'var(--sp-6)', textAlign: 'center' }}>
        No intents classified for this call.
      </p>
    );
  }

  const highConf = intents.filter(i => i.intent !== 'unknown' || i.confidence >= 0.5);
  const lowConf = intents.filter(i => i.intent === 'unknown' && i.confidence < 0.5);

  return (
    <div>
      {/* Intent Distribution Chart */}
      <div style={{ marginBottom: 'var(--sp-3)' }}>
        <IntentDistribution intents={intents} />
      </div>

      {/* Intent list */}
      <div className="intent-list">
        {highConf.map((intent, i) => {
          const color = getIntentColor(intent.intent);
          const confPct = intent.confidence * 100;
          const barColor = confPct >= 80 ? 'var(--success)' : confPct >= 50 ? 'var(--warning)' : 'var(--danger)';
          return (
            <div className="intent-item" key={i} style={{ borderLeft: `3px solid ${color}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)', marginBottom: '2px' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color, fontWeight: 600 }}>
                  {intent.intent.replace(/_/g, ' ')}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-dim)' }}>
                  [{intent.speaker}]
                </span>
                <div style={{ flex: 1 }} />
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', width: '80px' }}>
                  <div style={{ flex: 1, height: '4px', background: 'var(--bg)', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ width: `${confPct}%`, height: '100%', background: barColor, borderRadius: '2px' }} />
                  </div>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: barColor, width: '28px', textAlign: 'right' }}>
                    {confPct.toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="intent-text">&ldquo;{intent.utterance}&rdquo;</div>
            </div>
          );
        })}
      </div>
      {lowConf.length > 0 && (
        <details style={{ marginTop: 'var(--sp-2)' }}>
          <summary style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-dim)', cursor: 'pointer', padding: 'var(--sp-1)' }}>
            Low confidence ({lowConf.length} unknown)
          </summary>
          <div className="intent-list" style={{ opacity: 0.6 }}>
            {lowConf.map((intent, i) => (
              <div className="intent-item" key={i} style={{ borderLeft: '3px solid #6b7280' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: '#6b7280' }}>unknown</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-dim)' }}>[{intent.speaker}]</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--danger)', marginLeft: 'auto' }}>
                    {(intent.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="intent-text">&ldquo;{intent.utterance}&rdquo;</div>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
