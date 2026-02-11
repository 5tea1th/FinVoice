'use client';

import { CHART_COLORS } from './colors';

interface TooltipPayload {
  name: string;
  value: number;
  color?: string;
  payload?: Record<string, unknown>;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
  formatter?: (value: number, name: string) => string;
}

export default function ChartTooltip({ active, payload, label, formatter }: ChartTooltipProps) {
  if (!active || !payload?.length) return null;

  return (
    <div style={{
      background: CHART_COLORS.bgSurface,
      border: `1px solid ${CHART_COLORS.border}`,
      borderRadius: 6,
      padding: '8px 12px',
      fontFamily: 'var(--font-mono)',
      fontSize: '0.75rem',
      color: CHART_COLORS.text,
      boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
    }}>
      {label && (
        <div style={{ color: CHART_COLORS.textDim, marginBottom: 4, fontSize: '0.7rem' }}>
          {label}
        </div>
      )}
      {payload.map((entry, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: i > 0 ? 2 : 0 }}>
          <span style={{
            width: 8, height: 8, borderRadius: 2,
            background: entry.color || CHART_COLORS.s1,
            flexShrink: 0,
          }} />
          <span style={{ color: CHART_COLORS.textDim }}>{entry.name}:</span>
          <span style={{ fontWeight: 600 }}>
            {formatter ? formatter(entry.value, entry.name) : entry.value}
          </span>
        </div>
      ))}
    </div>
  );
}
