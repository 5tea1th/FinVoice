'use client';

import { useState } from 'react';

interface CorrectionsTabProps {
  callId: string;
}

export default function CorrectionsTab({ callId }: CorrectionsTabProps) {
  const [correctionNotes, setCorrectionNotes] = useState('');
  const [correctionSaved, setCorrectionSaved] = useState(false);

  const handleSubmitCorrections = async () => {
    try {
      const res = await fetch(`/api/calls/${callId}/corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: correctionNotes }),
      });
      if (res.ok) {
        setCorrectionSaved(true);
        setTimeout(() => setCorrectionSaved(false), 3000);
      }
    } catch { /* ignore */ }
  };

  return (
    <div style={{ padding: 'var(--sp-3)' }}>
      <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-3)' }}>Transcript Corrections & Notes</h4>
      <textarea
        value={correctionNotes}
        onChange={(e) => setCorrectionNotes(e.target.value)}
        placeholder="Add corrections, notes, or entity overrides..."
        style={{
          width: '100%',
          minHeight: '120px',
          padding: 'var(--sp-3)',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.85rem',
          color: 'var(--text)',
          resize: 'vertical',
        }}
      />
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-3)', marginTop: 'var(--sp-3)' }}>
        <button
          className="btn"
          onClick={handleSubmitCorrections}
          style={{ fontFamily: 'var(--font-mono)', background: 'var(--accent)', color: 'var(--bg)', padding: 'var(--sp-2) var(--sp-4)', border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer' }}
        >
          Save Corrections
        </button>
        {correctionSaved && (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--success)' }}>Saved!</span>
        )}
      </div>
    </div>
  );
}
