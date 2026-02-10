'use client';

import { useState } from 'react';
import {
  downloadExport,
  getExportLabel,
  type ExportFormat,
} from '@/lib/api';

const DATA_EXPORTS: ExportFormat[] = ['csv', 'parquet', 'jsonl'];
const TRAINING_EXPORTS: ExportFormat[] = [
  'training_intents',
  'training_sentiment',
  'training_entities',
];

const FORMAT_DESCRIPTIONS: Record<ExportFormat, string> = {
  csv: 'Flat summary per call — open in Excel or Google Sheets',
  parquet: 'Columnar ML-ready format — for pandas, Spark, Polars',
  jsonl: 'One full record per line — for HuggingFace datasets',
  training_intents: '(text, intent) pairs for intent classifier fine-tuning',
  training_sentiment: '(text, sentiment) pairs for sentiment model training',
  training_entities: '(text, entity_annotations) for NER model training',
};

const FORMAT_ICONS: Record<ExportFormat, string> = {
  csv: '\u{1F4CA}',
  parquet: '\u{1F9F1}',
  jsonl: '\u{1F4C4}',
  training_intents: '\u{1F3AF}',
  training_sentiment: '\u{1F4AC}',
  training_entities: '\u{1F50D}',
};

export default function ExportsView() {
  const [downloading, setDownloading] = useState<string | null>(null);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

  const handleDownload = async (format: ExportFormat) => {
    setDownloading(format);
    setMessage(null);
    const success = await downloadExport(format);
    setDownloading(null);
    if (success) {
      setMessage({ text: `${getExportLabel(format)} downloaded`, type: 'success' });
    } else {
      setMessage({ text: `Failed to download ${getExportLabel(format)}. Are there processed calls?`, type: 'error' });
    }
    setTimeout(() => setMessage(null), 4000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-6)' }}>
      {/* Status message */}
      {message && (
        <div
          style={{
            padding: 'var(--sp-3) var(--sp-4)',
            borderRadius: 'var(--r-md)',
            background: message.type === 'success' ? 'var(--success)' : 'var(--danger)',
            color: 'white',
            fontSize: '0.875rem',
            fontFamily: 'var(--font-mono)',
          }}
        >
          {message.text}
        </div>
      )}

      {/* Data Exports */}
      <div className="panel">
        <div className="panel-header">
          <h3 style={{ fontFamily: 'var(--font-mono)' }}>Data Exports</h3>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
            Export all processed call data
          </span>
        </div>
        <div style={{ padding: 'var(--sp-4)', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--sp-4)' }}>
          {DATA_EXPORTS.map(format => (
            <ExportCard
              key={format}
              format={format}
              downloading={downloading === format}
              onDownload={() => handleDownload(format)}
            />
          ))}
        </div>
      </div>

      {/* Training Data Exports */}
      <div className="panel">
        <div className="panel-header">
          <h3 style={{ fontFamily: 'var(--font-mono)' }}>Training Data</h3>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
            Generate fine-tuning datasets from processed calls
          </span>
        </div>
        <div style={{ padding: 'var(--sp-4)', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--sp-4)' }}>
          {TRAINING_EXPORTS.map(format => (
            <ExportCard
              key={format}
              format={format}
              downloading={downloading === format}
              onDownload={() => handleDownload(format)}
            />
          ))}
        </div>
      </div>

      {/* Info section */}
      <div className="panel">
        <div className="panel-header">
          <h3 style={{ fontFamily: 'var(--font-mono)' }}>Export Info</h3>
        </div>
        <div style={{ padding: 'var(--sp-4)', fontSize: '0.8125rem', color: 'var(--text-dim)', lineHeight: 1.7 }}>
          <p>All exports include every processed call in the system. Individual call downloads are available from the call detail view.</p>
          <p style={{ marginTop: 'var(--sp-2)' }}>
            Training data exports generate JSONL files suitable for HuggingFace datasets and model fine-tuning pipelines.
            Each processed call contributes labeled training pairs automatically.
          </p>
          <p style={{ marginTop: 'var(--sp-2)' }}>
            Masked transcript exports (with PII redacted) are available per-call from the call detail view.
          </p>
        </div>
      </div>
    </div>
  );
}

function ExportCard({
  format,
  downloading,
  onDownload,
}: {
  format: ExportFormat;
  downloading: boolean;
  onDownload: () => void;
}) {
  return (
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: 'var(--r-md)',
        padding: 'var(--sp-4)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--sp-3)',
        background: 'var(--surface)',
        transition: 'border-color 0.2s',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
        <span style={{ fontSize: '1.25rem' }}>{FORMAT_ICONS[format]}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: '0.875rem' }}>
          {getExportLabel(format)}
        </span>
      </div>
      <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', lineHeight: 1.5 }}>
        {FORMAT_DESCRIPTIONS[format]}
      </p>
      <button
        onClick={onDownload}
        disabled={downloading}
        style={{
          padding: 'var(--sp-2) var(--sp-4)',
          borderRadius: 'var(--r-sm)',
          border: '1px solid var(--border)',
          background: downloading ? 'var(--surface-2)' : 'var(--accent)',
          color: downloading ? 'var(--text-dim)' : 'white',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.8125rem',
          cursor: downloading ? 'wait' : 'pointer',
          transition: 'all 0.2s',
          alignSelf: 'flex-start',
        }}
      >
        {downloading ? 'Downloading...' : '\u2913 Download'}
      </button>
    </div>
  );
}
