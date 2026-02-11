// Hex color constants matching CSS variables from globals.css
// Recharts needs raw hex values, not CSS var() references

export const CHART_COLORS = {
  // Pipeline stage colors
  s1: '#6366f1', // indigo
  s2: '#10b981', // emerald
  s3: '#f59e0b', // amber
  s4: '#ef4444', // red
  s5: '#a855f7', // purple

  // Semantic
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  info: '#3b82f6',

  // Risk levels
  riskLow: '#10b981',
  riskMedium: '#f59e0b',
  riskHigh: '#ef4444',
  riskCritical: '#dc2626',

  // Emotions
  neutral: '#94a3b8',
  happy: '#10b981',
  angry: '#ef4444',
  sad: '#3b82f6',
  fearful: '#a855f7',
  disgusted: '#f97316',
  surprised: '#06b6d4',
  other: '#6b7280',

  // Intent categories
  positive: '#10b981',  // agreement, consent_given, acknowledgment
  negative: '#ef4444',  // refusal, complaint, dispute
  neutral_intent: '#94a3b8', // info_request, procedural
  financial: '#6366f1',  // financial_disclosure, guidance_forecast
  action: '#f59e0b',    // payment_promise, negotiation, escalation

  // UI
  text: '#E8ECF1',
  textDim: '#8B95A5',
  bg: '#0F1419',
  bgSurface: '#171D24',
  border: '#2A3441',
  gridLine: 'rgba(255,255,255,0.06)',
};

// Map intent names to color categories
export function getIntentColor(intent: string): string {
  const map: Record<string, string> = {
    agreement: CHART_COLORS.positive,
    consent_given: CHART_COLORS.positive,
    acknowledgment: CHART_COLORS.positive,
    greeting: CHART_COLORS.positive,
    payment_promise: CHART_COLORS.action,
    negotiation: CHART_COLORS.action,
    escalation: CHART_COLORS.danger,
    request_extension: CHART_COLORS.action,
    refusal: CHART_COLORS.negative,
    consent_denied: CHART_COLORS.negative,
    complaint: CHART_COLORS.negative,
    dispute: CHART_COLORS.negative,
    info_request: CHART_COLORS.info,
    question: CHART_COLORS.info,
    explanation: CHART_COLORS.info,
    procedural: CHART_COLORS.neutral_intent,
    financial_disclosure: CHART_COLORS.financial,
    guidance_forecast: CHART_COLORS.s5,
    risk_warning: CHART_COLORS.warning,
    unknown: '#4b5563',
  };
  return map[intent] || CHART_COLORS.neutral_intent;
}

export function getEmotionColor(emotion: string): string {
  const map: Record<string, string> = {
    neutral: CHART_COLORS.neutral,
    happy: CHART_COLORS.happy,
    angry: CHART_COLORS.angry,
    sad: CHART_COLORS.sad,
    fearful: CHART_COLORS.fearful,
    disgusted: CHART_COLORS.disgusted,
    surprised: CHART_COLORS.surprised,
    other: CHART_COLORS.other,
    '<unk>': '#4b5563',
  };
  return map[emotion] || CHART_COLORS.other;
}
