'use client';

import { PipelineWaterfall } from '@/components/charts';

interface PipelineTabProps {
  pipelineTimings: Record<string, number>;
  durationSec: number;
  audioQualityScore: number;
  audioQualityFlag: string;
  snrDb: number;
  speechPercentage: number;
  audioQualityComponents: { snr: number; clipping: number; speech_ratio: number; spectral: number };
  transcriptLength: number;
  wordCount: number;
  numSpeakers: number;
  overallTranscriptConfidence: number;
  numLowConfidenceSegments: number;
  entityCount: number;
  piiCount: number;
  obligationCount: number;
  compliancePassed: number;
  complianceTotal: number;
  fraudCount: number;
  tamperCount: number;
  tamperRisk: string;
  emotionSegments: number;
  dominantEmotion: string;
  intentCount: number;
  hasSummary: boolean;
  outcomeCount: number;
  actionCount: number;
}

export default function PipelineTab(props: PipelineTabProps) {
  const timingEntries = Object.entries(props.pipelineTimings);

  return (
    <div style={{ padding: 'var(--sp-2)' }}>
      {/* Waterfall chart */}
      {timingEntries.length > 0 && (
        <div style={{ marginBottom: 'var(--sp-4)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', marginBottom: 'var(--sp-2)', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Stage Timing</h4>
          <PipelineWaterfall timings={props.pipelineTimings} />
        </div>
      )}

      <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-3)', fontSize: '0.85rem' }}>Pipeline Stage Details</h4>

      {[
        {
          name: 'Stage 1: Audio Normalization',
          tools: 'FFmpeg',
          outputs: [`16kHz mono WAV (${props.durationSec.toFixed(0)}s)`],
        },
        {
          name: 'Stage 2: Audio Quality Analysis',
          tools: 'Silero VAD, openSMILE',
          outputs: [
            `Score: ${props.audioQualityScore}/100 (${props.audioQualityFlag})`,
            `SNR: ${props.snrDb.toFixed(1)} dB`,
            `Speech: ${props.speechPercentage.toFixed(0)}%`,
            `Components: SNR=${props.audioQualityComponents.snr}, Clip=${props.audioQualityComponents.clipping}, Speech=${props.audioQualityComponents.speech_ratio}, Spectral=${props.audioQualityComponents.spectral}`,
          ],
        },
        {
          name: 'Stage 2.5: Audio Cleanup',
          tools: 'librosa, Silero VAD',
          outputs: ['Dead air removal', 'Hold music detection', 'Speech stitching'],
        },
        {
          name: 'Stage 3: Transcription & Diarization',
          tools: 'WhisperX large-v3-turbo, pyannote 3.1',
          outputs: [
            `${props.transcriptLength} segments, ${props.wordCount.toLocaleString()} words`,
            `${props.numSpeakers} speakers detected`,
            `Confidence: ${(props.overallTranscriptConfidence * 100).toFixed(0)}% avg`,
            props.numLowConfidenceSegments > 0 ? `${props.numLowConfidenceSegments} low-confidence segments` : 'All segments high confidence',
          ],
        },
        {
          name: 'Stage 4A-B: Sentiment & Entity Extraction',
          tools: 'FinBERT, spaCy, regex, fine-tuned NER',
          outputs: [
            `${props.entityCount} financial entities`,
            `${props.piiCount} PII entities`,
            `${props.obligationCount} obligations detected`,
          ],
        },
        {
          name: 'Stage 4C-D: Compliance & Fraud',
          tools: 'Rule engine, Parselmouth, emotion2vec',
          outputs: [
            `${props.compliancePassed}/${props.complianceTotal} checks passed`,
            `${props.fraudCount} fraud signals`,
            `${props.tamperCount} tamper signals (${props.tamperRisk} risk)`,
          ],
        },
        {
          name: 'Stage 4E-H: Emotion & PII Analysis',
          tools: 'emotion2vec, Presidio, Detoxify',
          outputs: [
            props.emotionSegments ? `${props.emotionSegments} emotion segments analyzed` : 'Emotion analysis pending',
            props.dominantEmotion ? `Dominant: ${props.dominantEmotion}` : '',
          ].filter(Boolean),
        },
        {
          name: 'Stage 4I: LLM Intelligence',
          tools: 'Qwen3 8B via Ollama',
          outputs: [
            `${props.intentCount} intents classified`,
            props.hasSummary ? 'Summary generated' : 'Summary pending',
            `${props.outcomeCount} outcomes, ${props.actionCount} actions`,
          ],
        },
      ].map((stage, i) => {
        const timing = timingEntries.find(([k]) => k.toLowerCase().includes(stage.name.split(':')[0].toLowerCase().replace('stage ', '')));
        return (
          <div key={i} style={{
            background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            padding: 'var(--sp-2) var(--sp-3)', marginBottom: 'var(--sp-2)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 600 }}>{stage.name}</span>
              {timing && (
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--accent-primary)', background: 'rgba(59,130,246,0.1)', padding: '1px 8px', borderRadius: '8px' }}>
                  {timing[1]}s
                </span>
              )}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', marginBottom: '2px' }}>
              Tools: {stage.tools}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {stage.outputs.map((o, j) => <div key={j}>{o}</div>)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
