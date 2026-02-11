'use client';

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts';
import { CHART_COLORS } from '@/components/charts/colors';
import ChartTooltip from '@/components/charts/ChartTooltip';
import type { FraudSignal, TamperSignal } from '@/lib/api';

interface FraudTabProps {
  fraud: FraudSignal[];
  tamperSignals: TamperSignal[];
  tamperRisk: string;
}

const RISK_COLORS: Record<string, string> = {
  none: CHART_COLORS.neutral,
  low: CHART_COLORS.success,
  medium: CHART_COLORS.warning,
  high: CHART_COLORS.danger,
};

export default function FraudTab({ fraud, tamperSignals, tamperRisk }: FraudTabProps) {
  const tamperBadge = tamperRisk === 'high' ? 'badge-danger' : tamperRisk === 'medium' ? 'badge-warning' : 'badge-success';

  // Confidence chart data
  const chartData = fraud.length > 0
    ? fraud.map((f, i) => ({
        name: f.signal.substring(0, 16),
        confidence: Math.round(
          f.risk === 'high' ? 85 : f.risk === 'medium' ? 55 : f.risk === 'low' ? 25 : 5
        ),
        risk: f.risk,
        idx: i,
      }))
    : [];

  return (
    <div>
      {/* Confidence chart for multiple signals */}
      {chartData.length > 1 && (
        <div style={{ marginBottom: 'var(--sp-3)' }}>
          <div style={{ width: '100%', height: Math.max(100, chartData.length * 30) }}>
            <ResponsiveContainer>
              <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 80 }}>
                <XAxis type="number" domain={[0, 100]} tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={{ stroke: CHART_COLORS.border }} tickLine={false} unit="%" />
                <YAxis type="category" dataKey="name" tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} width={80} />
                <Tooltip content={<ChartTooltip formatter={(v) => `${v}% confidence`} />} />
                <Bar dataKey="confidence" radius={[0, 4, 4, 0]} maxBarSize={20}>
                  {chartData.map((entry) => (
                    <Cell key={entry.idx} fill={RISK_COLORS[entry.risk] || CHART_COLORS.neutral} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="fraud-list">
        {fraud.length === 0 ? (
          <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', padding: 'var(--sp-6)', textAlign: 'center' }}>
            No fraud signals detected — call appears clean.
          </p>
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
  );
}
