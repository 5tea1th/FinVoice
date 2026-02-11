'use client';

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts';
import { CHART_COLORS } from './colors';
import ChartTooltip from './ChartTooltip';

interface PipelineWaterfallProps {
  timings: Record<string, number>;
}

const STAGE_COLORS = [
  '#3b82f6', '#06b6d4', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6',
  '#6366f1', '#f97316',
];

export default function PipelineWaterfall({ timings }: PipelineWaterfallProps) {
  const entries = Object.entries(timings);
  if (entries.length === 0) return null;

  const data = entries.map(([name, seconds]) => ({
    name: name.replace(/^Stage \d+: /, '').substring(0, 14),
    fullName: name,
    seconds: Math.round(seconds * 10) / 10,
  }));

  return (
    <div style={{ width: '100%', height: Math.max(160, data.length * 28) }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 90 }}>
          <XAxis
            type="number"
            tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={{ stroke: CHART_COLORS.border }}
            tickLine={false}
            unit="s"
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={false}
            tickLine={false}
            width={90}
          />
          <Tooltip content={<ChartTooltip formatter={(v, n) => `${v}s — ${n}`} />} />
          <Bar dataKey="seconds" radius={[0, 4, 4, 0]} maxBarSize={22}>
            {data.map((_, i) => (
              <Cell key={i} fill={STAGE_COLORS[i % STAGE_COLORS.length]} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
