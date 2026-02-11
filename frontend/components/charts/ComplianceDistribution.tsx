'use client';

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts';
import { CHART_COLORS } from './colors';
import ChartTooltip from './ChartTooltip';

interface ComplianceDistributionProps {
  calls: Array<{ complianceScore: number }>;
}

const BIN_COLORS = [
  CHART_COLORS.danger,    // 0-20
  '#f97316',              // 20-40 orange
  CHART_COLORS.warning,   // 40-60
  '#84cc16',              // 60-80 lime
  CHART_COLORS.success,   // 80-100
];

export default function ComplianceDistribution({ calls }: ComplianceDistributionProps) {
  if (calls.length === 0) return null;

  const bins = [
    { range: '0-20', count: 0 },
    { range: '20-40', count: 0 },
    { range: '40-60', count: 0 },
    { range: '60-80', count: 0 },
    { range: '80-100', count: 0 },
  ];

  calls.forEach(c => {
    const score = c.complianceScore;
    if (score < 20) bins[0].count++;
    else if (score < 40) bins[1].count++;
    else if (score < 60) bins[2].count++;
    else if (score < 80) bins[3].count++;
    else bins[4].count++;
  });

  return (
    <div style={{ width: '100%', height: 220 }}>
      <ResponsiveContainer>
        <BarChart data={bins} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
          <XAxis
            dataKey="range"
            tick={{ fill: CHART_COLORS.textDim, fontSize: 11, fontFamily: 'var(--font-mono)' }}
            axisLine={{ stroke: CHART_COLORS.border }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={false}
            tickLine={false}
            allowDecimals={false}
          />
          <Tooltip content={<ChartTooltip formatter={(v) => `${v} calls`} />} />
          <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={40}>
            {bins.map((_, i) => (
              <Cell key={i} fill={BIN_COLORS[i]} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
