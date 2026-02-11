'use client';

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts';
import { CHART_COLORS, getIntentColor } from './colors';
import ChartTooltip from './ChartTooltip';

interface IntentDistributionProps {
  intents: Array<{ intent: string }>;
}

export default function IntentDistribution({ intents }: IntentDistributionProps) {
  if (intents.length === 0) return null;

  // Count intents
  const counts: Record<string, number> = {};
  intents.forEach(i => {
    counts[i.intent] = (counts[i.intent] || 0) + 1;
  });

  const data = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([name, count]) => ({
      name: name.replace(/_/g, ' '),
      rawName: name,
      count,
    }));

  return (
    <div style={{ width: '100%', height: Math.max(180, data.length * 32) }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 80 }}>
          <XAxis
            type="number"
            tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={{ stroke: CHART_COLORS.border }}
            tickLine={false}
            allowDecimals={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: CHART_COLORS.textDim, fontSize: 11, fontFamily: 'var(--font-mono)' }}
            axisLine={false}
            tickLine={false}
            width={80}
          />
          <Tooltip content={<ChartTooltip formatter={(v) => `${v} utterances`} />} />
          <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={24}>
            {data.map((entry) => (
              <Cell key={entry.rawName} fill={getIntentColor(entry.rawName)} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
