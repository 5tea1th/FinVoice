'use client';

import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { CHART_COLORS, getEmotionColor } from './colors';
import ChartTooltip from './ChartTooltip';

interface EmotionTimelineProps {
  segmentEmotions: Array<{
    segment_id: number;
    speaker: string;
    emotion: string;
    score: number;
  }>;
}

const EMOTION_ORDER = ['happy', 'neutral', 'sad', 'angry', 'fearful', 'surprised', 'disgusted'];

export default function EmotionTimeline({ segmentEmotions }: EmotionTimelineProps) {
  if (segmentEmotions.length === 0) return null;

  // Collect all present emotions
  const presentEmotions = new Set(segmentEmotions.map(s => s.emotion).filter(e => e !== '<unk>' && e !== 'other'));
  const emotions = EMOTION_ORDER.filter(e => presentEmotions.has(e));

  // Group segments into windows of ~3 for smoother visualization
  const windowSize = Math.max(1, Math.floor(segmentEmotions.length / 30));
  const windows: Array<Record<string, number>> = [];

  for (let i = 0; i < segmentEmotions.length; i += windowSize) {
    const window = segmentEmotions.slice(i, i + windowSize);
    const counts: Record<string, number> = { segment: Math.floor(i / windowSize) + 1 };
    const total = window.length;
    emotions.forEach(e => { counts[e] = 0; });
    window.forEach(seg => {
      if (emotions.includes(seg.emotion)) {
        counts[seg.emotion] = (counts[seg.emotion] || 0) + 1;
      }
    });
    // Normalize to percentages
    emotions.forEach(e => { counts[e] = Math.round((counts[e] / total) * 100); });
    windows.push(counts);
  }

  return (
    <div style={{ width: '100%', height: 200 }}>
      <ResponsiveContainer>
        <AreaChart data={windows} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
          <XAxis
            dataKey="segment"
            tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={{ stroke: CHART_COLORS.border }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<ChartTooltip formatter={(v) => `${v}%`} />} />
          {emotions.map((emotion) => (
            <Area
              key={emotion}
              type="monotone"
              dataKey={emotion}
              stackId="1"
              stroke={getEmotionColor(emotion)}
              fill={getEmotionColor(emotion)}
              fillOpacity={0.6}
              strokeWidth={0}
              name={emotion}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
