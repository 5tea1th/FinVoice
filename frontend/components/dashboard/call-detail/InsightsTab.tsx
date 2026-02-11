'use client';

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis } from 'recharts';
import { CHART_COLORS } from '@/components/charts/colors';
import ChartTooltip from '@/components/charts/ChartTooltip';
import type { FinancialInsights } from '@/lib/api';

interface InsightsTabProps {
  financialInsights: FinancialInsights;
  onSegmentClick?: (segmentId: number) => void;
}

export default function InsightsTab({ financialInsights: fi, onSegmentClick }: InsightsTabProps) {
  const eff = fi?.call_effectiveness;

  const metricColor = (type: string) => {
    switch (type) {
      case 'financial_amount': return '#22c55e';
      case 'rate': return '#f59e0b';
      case 'organization': return '#8b5cf6';
      case 'key_person': return '#06b6d4';
      case 'date': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  return (
    <div style={{ padding: 'var(--sp-2)' }}>
      {/* Discussion Topics as bar chart */}
      {fi?.discussion_topics?.length > 0 && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', marginBottom: 'var(--sp-2)', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Discussion Topics</h4>
          <div style={{ width: '100%', height: Math.max(140, fi.discussion_topics.length * 30) }}>
            <ResponsiveContainer>
              <BarChart
                data={fi.discussion_topics.map(t => ({ name: t.topic, pct: t.percentage, mentions: t.mentions }))}
                layout="vertical"
                margin={{ top: 4, right: 8, bottom: 4, left: 120 }}
              >
                <XAxis type="number" tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={{ stroke: CHART_COLORS.border }} tickLine={false} unit="%" />
                <YAxis type="category" dataKey="name" tick={{ fill: CHART_COLORS.textDim, fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} width={120} />
                <Tooltip content={<ChartTooltip formatter={(v) => `${v}%`} />} />
                <Bar dataKey="pct" radius={[0, 4, 4, 0]} maxBarSize={22}>
                  {fi.discussion_topics.map((_, i) => {
                    const colors = [CHART_COLORS.s1, CHART_COLORS.s2, CHART_COLORS.s3, CHART_COLORS.s5, CHART_COLORS.info, CHART_COLORS.warning];
                    return <Cell key={i} fill={colors[i % colors.length]} fillOpacity={0.85} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Topic Sentiment */}
      {fi?.topic_sentiment && Object.keys(fi.topic_sentiment).length > 0 && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', marginBottom: 'var(--sp-2)', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Topic Sentiment</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {Object.entries(fi.topic_sentiment)
              .sort(([, a], [, b]) => Math.abs(b as number) - Math.abs(a as number))
              .map(([topic, score]) => {
              const s = score as number;
              const color = s > 0.15 ? CHART_COLORS.success : s < -0.15 ? CHART_COLORS.danger : CHART_COLORS.warning;
              const label = s > 0.15 ? 'Positive' : s < -0.15 ? 'Negative' : 'Neutral';
              return (
                <div key={topic} style={{
                  display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px',
                  background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                  borderLeft: `3px solid ${color}`,
                }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 600 }}>{topic}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color, fontWeight: 600 }}>{label} ({s > 0 ? '+' : ''}{s.toFixed(2)})</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Key Metrics */}
      {fi?.key_metrics?.length > 0 && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', marginBottom: 'var(--sp-2)', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Key Entities & Metrics</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 'var(--sp-2)' }}>
            {fi.key_metrics.map((m, i) => {
              const tc = metricColor(m.type);
              const clickable = m.segment_id != null && onSegmentClick;
              return (
                <div key={i}
                  onClick={clickable ? () => onSegmentClick(m.segment_id!) : undefined}
                  style={{
                    padding: 'var(--sp-2) var(--sp-3)', background: 'var(--bg-surface)', border: '1px solid var(--border)', borderLeft: `3px solid ${tc}`, borderRadius: 'var(--radius-sm)',
                    ...(clickable ? { cursor: 'pointer', transition: 'background 0.15s, border-color 0.15s' } : {}),
                  }}
                  onMouseEnter={clickable ? (e) => { e.currentTarget.style.background = 'var(--bg-hover)'; e.currentTarget.style.borderColor = tc; } : undefined}
                  onMouseLeave={clickable ? (e) => { e.currentTarget.style.background = 'var(--bg-surface)'; e.currentTarget.style.borderColor = 'var(--border)'; } : undefined}
                >
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: tc, textTransform: 'uppercase', marginBottom: '2px' }}>{m.type.replace(/_/g, ' ')}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', fontWeight: 700 }}>{m.value}</div>
                  {m.context && m.context !== m.value && <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--text-dim)', fontStyle: 'italic', marginTop: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>&ldquo;{m.context}&rdquo;</div>}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Risk Factors */}
      {fi?.risk_factors?.length > 0 && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', marginBottom: 'var(--sp-2)', color: 'var(--danger)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Risk Factors</h4>
          {fi.risk_factors.map((r, i) => (
            <div key={i} style={{ display: 'flex', gap: 'var(--sp-2)', padding: 'var(--sp-2)', marginBottom: '4px', background: 'rgba(239,68,68,0.05)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--danger)' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--danger)', textTransform: 'uppercase', minWidth: '100px', flexShrink: 0 }}>{r.type.replace(/_/g, ' ')}</span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{r.detail}</span>
            </div>
          ))}
        </div>
      )}

      {/* Recommendations */}
      {fi?.recommendations?.length > 0 && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', marginBottom: 'var(--sp-2)', color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Recommendations</h4>
          {fi.recommendations.map((r, i) => (
            <div key={i} style={{ padding: 'var(--sp-2) var(--sp-3)', marginBottom: '4px', background: 'rgba(59,130,246,0.05)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--accent-primary)', fontSize: '0.8rem' }}>
              {r}
            </div>
          ))}
        </div>
      )}

      {/* Call Effectiveness Radar */}
      {eff && (
        <div>
          <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', marginBottom: 'var(--sp-2)', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Call Analysis Stats</h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--sp-3)' }}>
            <div style={{ height: 220 }}>
              <ResponsiveContainer>
                <RadarChart data={[
                  { label: 'Segments', value: Math.min(100, eff.total_segments) },
                  { label: 'Speakers', value: eff.total_speakers * 25 },
                  { label: 'Entities', value: Math.min(100, eff.entities_extracted * 5) },
                  { label: 'Intents', value: Math.min(100, eff.intents_classified) },
                  { label: 'Disclosure', value: eff.disclosure_rate },
                  { label: 'Q&A', value: eff.qa_rate },
                ]} cx="50%" cy="50%" outerRadius="65%">
                  <PolarGrid stroke={CHART_COLORS.border} />
                  <PolarAngleAxis dataKey="label" tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }} />
                  <Radar dataKey="value" stroke={CHART_COLORS.s1} fill={CHART_COLORS.s1} fillOpacity={0.2} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--sp-2)', alignContent: 'center' }}>
              {[
                { label: 'Segments', value: eff.total_segments },
                { label: 'Speakers', value: eff.total_speakers },
                { label: 'Entities', value: eff.entities_extracted },
                { label: 'Intents', value: eff.intents_classified },
                { label: 'Words', value: eff.total_words?.toLocaleString() ?? '—' },
                { label: 'Avg/Seg', value: eff.avg_words_per_segment ?? '—' },
                { label: 'Unknown %', value: `${eff.unknown_rate}%`, warn: eff.unknown_rate > 20 },
                { label: 'Balance %', value: `${eff.speaker_balance ?? 0}%` },
              ].map((s, i) => (
                <div key={i} style={{ padding: 'var(--sp-2)', background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 700, color: s.warn ? 'var(--danger)' : 'var(--text)' }}>{s.value}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {(!fi || (!fi.key_metrics?.length && !fi.discussion_topics?.length)) && (
        <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', padding: 'var(--sp-6)', textAlign: 'center' }}>
          Financial insights not available — process a new call to generate insights.
        </p>
      )}
    </div>
  );
}
