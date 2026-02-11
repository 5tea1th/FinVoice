'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { CHART_COLORS } from './colors';
import ChartTooltip from './ChartTooltip';

interface RiskDonutProps {
  distribution: Record<string, number>;
}

const RISK_COLORS: Record<string, string> = {
  low: CHART_COLORS.riskLow,
  medium: CHART_COLORS.riskMedium,
  high: CHART_COLORS.riskHigh,
  critical: CHART_COLORS.riskCritical,
};

export default function RiskDonut({ distribution }: RiskDonutProps) {
  const data = Object.entries(distribution)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }));

  if (data.length === 0) return null;

  const total = data.reduce((s, d) => s + d.value, 0);

  return (
    <div style={{ width: '100%', height: 220, position: 'relative' }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={55}
            outerRadius={80}
            paddingAngle={3}
            dataKey="value"
            stroke="none"
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={RISK_COLORS[entry.name] || CHART_COLORS.neutral} />
            ))}
          </Pie>
          <Tooltip content={<ChartTooltip formatter={(v, n) => `${v} calls (${((v / total) * 100).toFixed(0)}%) — ${n}`} />} />
        </PieChart>
      </ResponsiveContainer>
      {/* Center label */}
      <div style={{
        position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
        textAlign: 'center', pointerEvents: 'none',
      }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.5rem', fontWeight: 700, color: CHART_COLORS.text }}>{total}</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: CHART_COLORS.textDim, textTransform: 'uppercase', letterSpacing: '0.05em' }}>calls</div>
      </div>
    </div>
  );
}
