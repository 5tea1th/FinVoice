'use client';

import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Legend } from 'recharts';
import { CHART_COLORS, getEmotionColor } from './colors';

interface EmotionRadarProps {
  speakerBreakdown: Record<string, { distribution: Record<string, number> }>;
}

const EMOTIONS_ORDER = ['happy', 'neutral', 'sad', 'angry', 'fearful', 'surprised', 'disgusted'];

export default function EmotionRadar({ speakerBreakdown }: EmotionRadarProps) {
  const speakers = Object.keys(speakerBreakdown);
  if (speakers.length === 0) return null;

  // Collect all emotions present
  const allEmotions = new Set<string>();
  speakers.forEach(s => {
    Object.keys(speakerBreakdown[s].distribution).forEach(e => {
      if (e !== '<unk>' && e !== 'other') allEmotions.add(e);
    });
  });
  const emotions = EMOTIONS_ORDER.filter(e => allEmotions.has(e));
  if (emotions.length < 3) return null;

  const data = emotions.map(emotion => {
    const entry: Record<string, string | number> = { emotion };
    speakers.forEach(speaker => {
      entry[speaker] = Math.round((speakerBreakdown[speaker].distribution[emotion] || 0) * 100);
    });
    return entry;
  });

  const SPEAKER_COLORS = [CHART_COLORS.s1, CHART_COLORS.success, CHART_COLORS.s3, CHART_COLORS.s5];

  return (
    <div style={{ width: '100%', height: 260 }}>
      <ResponsiveContainer>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
          <PolarGrid stroke={CHART_COLORS.border} />
          <PolarAngleAxis
            dataKey="emotion"
            tick={(props: Record<string, unknown>) => {
              const { x, y, payload } = props as { x: number; y: number; payload: { value: string } };
              return (
                <text
                  x={x} y={y}
                  fill={getEmotionColor(payload.value)}
                  fontSize={11}
                  fontFamily="var(--font-mono)"
                  textAnchor="middle"
                  dominantBaseline="central"
                >
                  {payload.value}
                </text>
              );
            }}
          />
          {speakers.map((speaker, i) => (
            <Radar
              key={speaker}
              name={speaker}
              dataKey={speaker}
              stroke={SPEAKER_COLORS[i % SPEAKER_COLORS.length]}
              fill={SPEAKER_COLORS[i % SPEAKER_COLORS.length]}
              fillOpacity={0.15}
              strokeWidth={2}
            />
          ))}
          <Legend
            wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: CHART_COLORS.textDim }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
