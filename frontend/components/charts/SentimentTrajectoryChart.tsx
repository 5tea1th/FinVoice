'use client';

import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine } from 'recharts';
import { CHART_COLORS } from './colors';
import ChartTooltip from './ChartTooltip';

interface SentimentTrajectoryChartProps {
  customerTrajectory: number[];
  agentTrajectory: number[];
}

export default function SentimentTrajectoryChart({ customerTrajectory, agentTrajectory }: SentimentTrajectoryChartProps) {
  const maxLen = Math.max(customerTrajectory.length, agentTrajectory.length);
  if (maxLen === 0) return null;

  const data = Array.from({ length: maxLen }, (_, i) => ({
    segment: i + 1,
    customer: customerTrajectory[i] ?? null,
    agent: agentTrajectory[i] ?? null,
  }));

  return (
    <div style={{ width: '100%', height: 220 }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
          <defs>
            <linearGradient id="gradCustomer" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART_COLORS.success} stopOpacity={0.3} />
              <stop offset="100%" stopColor={CHART_COLORS.success} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gradAgent" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART_COLORS.s1} stopOpacity={0.3} />
              <stop offset="100%" stopColor={CHART_COLORS.s1} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="segment"
            tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={{ stroke: CHART_COLORS.border }}
            tickLine={false}
          />
          <YAxis
            domain={[-1, 1]}
            tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={false}
            tickLine={false}
            ticks={[-1, -0.5, 0, 0.5, 1]}
          />
          <ReferenceLine y={0} stroke={CHART_COLORS.border} strokeDasharray="3 3" />
          <Tooltip content={<ChartTooltip formatter={(v) => v.toFixed(2)} />} />
          <Area
            type="monotone"
            dataKey="customer"
            stroke={CHART_COLORS.success}
            fill="url(#gradCustomer)"
            strokeWidth={2}
            dot={false}
            connectNulls
            name="Customer"
          />
          <Area
            type="monotone"
            dataKey="agent"
            stroke={CHART_COLORS.s1}
            fill="url(#gradAgent)"
            strokeWidth={2}
            dot={false}
            connectNulls
            name="Agent"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
