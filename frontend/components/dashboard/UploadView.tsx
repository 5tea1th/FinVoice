'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { getPipelineStages, uploadCall, getProgress, type PipelineProgress } from '@/lib/api';

function parseTimeInput(val: string): number {
  if (val.includes(':')) {
    const [m, s] = val.split(':').map(Number);
    return (m || 0) * 60 + (s || 0);
  }
  return Number(val) || 0;
}

function formatSeconds(sec: number): string {
  if (!sec) return '';
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

// ─── Waveform component — renders audio file as a waveform on canvas ───
function Waveform({ file, trimStart, trimEnd }: { file: File; trimStart: number; trimEnd: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [peaks, setPeaks] = useState<number[]>([]);
  const [duration, setDuration] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!file || file.size === 0) { setLoading(false); return; }
    let cancelled = false;
    const ac = new AudioContext();

    file.arrayBuffer().then(buf => ac.decodeAudioData(buf)).then(audioBuffer => {
      if (cancelled) return;
      const raw = audioBuffer.getChannelData(0);
      const numBars = 200;
      const blockSize = Math.floor(raw.length / numBars);
      const bars: number[] = [];
      for (let i = 0; i < numBars; i++) {
        let sum = 0;
        const start = i * blockSize;
        for (let j = start; j < start + blockSize && j < raw.length; j++) {
          sum += Math.abs(raw[j]);
        }
        bars.push(sum / blockSize);
      }
      // Normalize
      const max = Math.max(...bars, 0.001);
      setPeaks(bars.map(b => b / max));
      setDuration(audioBuffer.duration);
      setLoading(false);
      ac.close();
    }).catch(() => { setLoading(false); ac.close(); });

    return () => { cancelled = true; };
  }, [file]);

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || peaks.length === 0 || duration === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);
    const barW = w / peaks.length;
    const midY = h / 2;

    // Trim region markers
    const trimStartPx = trimStart > 0 ? (trimStart / duration) * w : 0;
    const trimEndPx = trimEnd > 0 ? (trimEnd / duration) * w : w;

    // Dim outside trim region
    if (trimStart > 0 || trimEnd > 0) {
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      if (trimStartPx > 0) ctx.fillRect(0, 0, trimStartPx, h);
      if (trimEndPx < w) ctx.fillRect(trimEndPx, 0, w - trimEndPx, h);
    }

    // Draw bars
    for (let i = 0; i < peaks.length; i++) {
      const x = i * barW;
      const barH = peaks[i] * midY * 0.9;
      const inTrim = x >= trimStartPx && x <= trimEndPx;
      ctx.fillStyle = inTrim ? 'var(--orange)' : 'rgba(100,116,139,0.3)';
      ctx.fillRect(x + 0.5, midY - barH, Math.max(barW - 1, 1), barH * 2);
    }

    // Trim boundary lines
    if (trimStart > 0) {
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 2]);
      ctx.beginPath(); ctx.moveTo(trimStartPx, 0); ctx.lineTo(trimStartPx, h); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (trimEnd > 0 && trimEndPx < w) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 2]);
      ctx.beginPath(); ctx.moveTo(trimEndPx, 0); ctx.lineTo(trimEndPx, h); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [peaks, duration, trimStart, trimEnd]);

  if (loading) {
    return (
      <div style={{ height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
        Loading waveform...
      </div>
    );
  }

  if (peaks.length === 0) return null;

  return (
    <div style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: 80, display: 'block', borderRadius: 4 }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
        <span>0:00</span>
        <span>{formatSeconds(duration / 4)}</span>
        <span>{formatSeconds(duration / 2)}</span>
        <span>{formatSeconds(duration * 3 / 4)}</span>
        <span>{formatSeconds(duration)}</span>
      </div>
    </div>
  );
}

interface UploadFile {
  file: File;
  name: string;
  progress: number;
  done: boolean;
  error: string | null;
  callId: string | null;
}

// Map backend internal stages to the 5 frontend pipeline stages (0-indexed)
// Pipeline execution order: 1→2→2.5→3→4A→4B→4C→4D→4E→4F→4G→4H→4I→5
// Must always progress forward — 4I runs last in stage 4, so it maps to frontend 4
function mapBackendStage(backendStage: string): number {
  switch (backendStage) {
    case '1': case '2': case '2.5': return 0;         // Ingestion
    case '3': return 1;                                 // Transcription
    case '4A': case '4B': return 2;                    // NLU & Extraction
    case '4C': case '4D': case '4E': case '4F':
    case '4G': case '4H': case '4I': return 3;        // Compliance & Fraud + LLM
    case '5': case 'done': return 4;                    // Output & Storage
    default: return -1;
  }
}

function computeStageStates(progress: PipelineProgress): ('idle' | 'active' | 'done')[] {
  const states: ('idle' | 'active' | 'done')[] = ['idle', 'idle', 'idle', 'idle', 'idle'];

  if (!progress.stages_completed?.length && !progress.current_stage) return states;

  // Mark completed frontend stages based on completed backend stages
  const completedFrontendStages = new Set<number>();
  for (const s of (progress.stages_completed || [])) {
    completedFrontendStages.add(mapBackendStage(s));
  }

  // Determine active frontend stage from current backend stage
  const activeFrontendStage = progress.current_stage ? mapBackendStage(progress.current_stage) : -1;

  for (let i = 0; i < 5; i++) {
    if (progress.status === 'complete') {
      states[i] = 'done';
    } else if (i === activeFrontendStage) {
      states[i] = 'active';
    } else if (completedFrontendStages.has(i) && i < activeFrontendStage) {
      states[i] = 'done';
    } else if (i < activeFrontendStage) {
      states[i] = 'done';
    }
  }

  return states;
}

function computeProgressPercent(progress: PipelineProgress): number {
  if (progress.status === 'complete') return 100;
  if (progress.status === 'error') return 0;
  const total = 13; // Total internal stages
  const completed = progress.stages_completed?.length || 0;
  return Math.min(Math.round((completed / total) * 100), 99);
}

// Persist active pipeline tracking in sessionStorage so navigation doesn't lose it
function saveActiveCall(callId: string, fileName: string) {
  if (typeof window !== 'undefined') {
    sessionStorage.setItem('finvoice_active_call', JSON.stringify({ callId, fileName }));
  }
}
function loadActiveCall(): { callId: string; fileName: string } | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = sessionStorage.getItem('finvoice_active_call');
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}
function clearActiveCall() {
  if (typeof window !== 'undefined') sessionStorage.removeItem('finvoice_active_call');
}

export default function UploadView() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragover, setDragover] = useState(false);
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [processing, setProcessing] = useState(false);
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [stageStates, setStageStates] = useState<('idle' | 'active' | 'done')[]>([]);
  const [currentStageName, setCurrentStageName] = useState('');
  const [liveData, setLiveData] = useState<PipelineProgress>({} as PipelineProgress);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeCallIdRef = useRef<string | null>(null);
  const resumedRef = useRef(false);

  // Helper: apply progress update to UI
  const applyProgress = useCallback((progress: PipelineProgress, fileIdx: number) => {
    if (progress.status === 'idle' || progress.status === 'unknown') return;

    setStageStates(computeStageStates(progress));
    setCurrentStageName(progress.current_stage_name || '');
    setLiveData(progress);
    const pct = computeProgressPercent(progress);
    setFiles(prev => prev.map((f, i) => i === fileIdx ? { ...f, progress: pct } : f));

    if (progress.status === 'complete') {
      if (pollingRef.current) clearInterval(pollingRef.current);
      pollingRef.current = null;
      activeCallIdRef.current = null;
      clearActiveCall();
      setProcessing(false);
      setFiles(prev => prev.map((f, i) =>
        i === fileIdx ? { ...f, progress: 100, done: true, callId: progress.call_id || '' } : f
      ));
    } else if (progress.status === 'error') {
      if (pollingRef.current) clearInterval(pollingRef.current);
      pollingRef.current = null;
      activeCallIdRef.current = null;
      clearActiveCall();
      setProcessing(false);
      setFiles(prev => prev.map((f, i) =>
        i === fileIdx ? { ...f, progress: 0, error: progress.error || 'Pipeline error' } : f
      ));
    }
  }, []);

  // On mount: resume tracking if there's an active pipeline from before navigation
  useEffect(() => {
    if (resumedRef.current) return;
    resumedRef.current = true;

    const active = loadActiveCall();
    if (!active) return;

    // Restore file entry and start polling
    setFiles([{
      file: new File([], active.fileName),
      name: active.fileName,
      progress: 1, // non-zero so it shows as processing
      done: false,
      error: null,
      callId: active.callId,
    }]);
    setProcessing(true);
    const stages = getPipelineStages();
    setStageStates(stages.map(() => 'idle'));
    setCurrentStageName('Resuming...');

    // Fetch current state immediately, then start polling
    getProgress(active.callId).then(progress => {
      if (progress.status === 'complete') {
        clearActiveCall();
        setProcessing(false);
        setStageStates(computeStageStates(progress));
        setFiles([{
          file: new File([], active.fileName),
          name: active.fileName,
          progress: 100,
          done: true,
          error: null,
          callId: progress.call_id || active.callId,
        }]);
      } else if (progress.status === 'error') {
        clearActiveCall();
        setProcessing(false);
        setFiles([{
          file: new File([], active.fileName),
          name: active.fileName,
          progress: 0,
          done: false,
          error: progress.error || 'Pipeline error',
          callId: active.callId,
        }]);
      } else if (progress.status === 'processing') {
        applyProgress(progress, 0);
        startPollingForCall(active.callId, 0);
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Tab visibility handler for browser tab switching
  useEffect(() => {
    const handleVisibility = () => {
      if (document.visibilityState === 'visible' && activeCallIdRef.current) {
        const callId = activeCallIdRef.current;
        const fileIdx = files.findIndex(f => f.callId === callId && !f.done && !f.error);
        if (fileIdx >= 0) {
          getProgress(callId).then(progress => applyProgress(progress, fileIdx));
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibility);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [files, applyProgress]);

  const startPollingForCall = useCallback((callId: string, fileIdx: number) => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    activeCallIdRef.current = callId;

    pollingRef.current = setInterval(async () => {
      const progress = await getProgress(callId);
      applyProgress(progress, fileIdx);
    }, 2000);
  }, [applyProgress]);

  const startPolling = useCallback((fileIdx: number, callId: string) => {
    startPollingForCall(callId, fileIdx);
  }, [startPollingForCall]);

  // Stage files without auto-processing — user clicks "Process" to start
  const handleFiles = useCallback((fileList: FileList) => {
    const newFiles = Array.from(fileList).map(f => ({
      file: f,
      name: f.name,
      progress: 0,
      done: false,
      error: null as string | null,
      callId: null as string | null,
    }));
    setFiles(prev => [...prev, ...newFiles]);
  }, []);

  const processFiles = useCallback(() => {
    // Find unprocessed files (progress 0, no error, no callId)
    const toProcess = files.filter(f => !f.done && !f.error && !f.callId && f.progress === 0);
    if (toProcess.length === 0) return;

    setProcessing(true);
    const stages = getPipelineStages();
    setStageStates(stages.map(() => 'idle'));
    setCurrentStageName('Starting pipeline...');

    toProcess.forEach(uf => {
      const fileIdx = files.indexOf(uf);
      const opts = {
        startTime: parseTimeInput(startTime),
        endTime: parseTimeInput(endTime),
      };
      uploadCall(uf.file, opts).then(result => {
        if (result.success && result.callId) {
          saveActiveCall(result.callId, uf.name);
          setFiles(prev => prev.map((f, i) =>
            i === fileIdx ? { ...f, callId: result.callId } : f
          ));
          startPolling(fileIdx, result.callId);
        } else {
          setFiles(prev => prev.map((f, i) =>
            i === fileIdx ? { ...f, progress: 0, error: result.message || 'Upload failed' } : f
          ));
        }
      });
    });
  }, [files, startPolling, startTime, endTime]);

  const hasStagedFiles = files.some(f => !f.done && !f.error && !f.callId && f.progress === 0);
  const stages = getPipelineStages();
  const stageColors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#a855f7'];

  return (
    <>
      <div
        className={`upload-zone${dragover ? ' dragover' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragover(true); }}
        onDragLeave={() => setDragover(false)}
        onDrop={e => { e.preventDefault(); setDragover(false); handleFiles(e.dataTransfer.files); }}
      >
        <div className="upload-icon">
          <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
            <path d="M8 40 Q16 20 24 35 Q28 42 32 28 Q36 14 40 30 Q44 46 48 25 Q52 10 56 40" stroke="var(--orange)" strokeWidth="2" fill="none" strokeLinecap="round" />
            <path d="M32 52 L32 38 M26 44 L32 38 L38 44" stroke="var(--text-dim)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <h3 style={{ fontFamily: 'var(--font-mono)', position: 'relative' }}>Drop audio files or click to browse</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-dim)', position: 'relative' }}>
          WAV, MP3, FLAC, OGG &middot; Up to 500MB
        </p>
        <div style={{ display: 'flex', gap: 'var(--sp-3)', justifyContent: 'center', marginTop: 'var(--sp-3)', position: 'relative' }}>
          {['Transcribe', 'Analyze', 'Comply', 'Export'].map((step, i) => (
            <span key={step} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: ['#6366f1', '#10b981', '#f59e0b', '#a855f7'][i], display: 'inline-block' }} />
              {step}
            </span>
          ))}
        </div>
        <input type="file" ref={fileInputRef} accept="audio/*" multiple hidden onChange={e => e.target.files && handleFiles(e.target.files)} />
      </div>

      {/* Staged files + trim options — shown when files selected but not yet processing */}
      {hasStagedFiles && (
        <div className="panel" style={{ marginTop: 'var(--sp-4)' }}>
          <div className="panel-header">
            <h3 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>Ready to process</h3>
          </div>
          <div style={{ padding: 'var(--sp-4)' }}>
            {/* Staged file list */}
            {files.filter(f => !f.done && !f.error && !f.callId).map((file, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-3)', padding: 'var(--sp-2) 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', flex: 1 }}>{file.name}</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                  {(file.file.size / (1024 * 1024)).toFixed(1)} MB
                </span>
                <button
                  onClick={() => setFiles(prev => prev.filter(f => f !== file))}
                  style={{
                    background: 'none', border: 'none', color: 'var(--text-dim)',
                    cursor: 'pointer', fontSize: '1.1rem', padding: '0 4px',
                  }}
                  title="Remove"
                >
                  x
                </button>
              </div>
            ))}

            {/* Trim options */}
            <div style={{ display: 'flex', gap: 'var(--sp-4)', alignItems: 'center', marginTop: 'var(--sp-4)', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                Trim (optional):
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>From</label>
                <input
                  type="text"
                  placeholder="0:00"
                  value={startTime}
                  onChange={e => setStartTime(e.target.value)}
                  style={{
                    width: '5rem', padding: '0.3rem 0.5rem', fontSize: '0.8rem',
                    fontFamily: 'var(--font-mono)', background: 'var(--bg-card)',
                    border: '1px solid var(--border)', borderRadius: '4px', color: 'var(--text)',
                  }}
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>To</label>
                <input
                  type="text"
                  placeholder="mm:ss"
                  value={endTime}
                  onChange={e => setEndTime(e.target.value)}
                  style={{
                    width: '5rem', padding: '0.3rem 0.5rem', fontSize: '0.8rem',
                    fontFamily: 'var(--font-mono)', background: 'var(--bg-card)',
                    border: '1px solid var(--border)', borderRadius: '4px', color: 'var(--text)',
                  }}
                />
              </div>
              {(startTime || endTime) && (
                <span style={{ fontSize: '0.7rem', color: 'var(--orange)', fontFamily: 'var(--font-mono)' }}>
                  {formatSeconds(parseTimeInput(startTime)) || '0:00'} → {formatSeconds(parseTimeInput(endTime)) || 'end'}
                </span>
              )}
            </div>

            {/* Process button */}
            <button
              onClick={processFiles}
              style={{
                marginTop: 'var(--sp-4)', padding: '0.75rem 2rem',
                fontFamily: 'var(--font-mono)', fontSize: '0.9rem', fontWeight: 600,
                background: 'linear-gradient(135deg, var(--orange), #f59e0b)', color: '#000', border: 'none',
                borderRadius: '8px', cursor: 'pointer', width: '100%',
                transition: 'all .2s', boxShadow: '0 4px 16px color-mix(in srgb, var(--orange) 30%, transparent)',
              }}
              onMouseOver={e => { (e.target as HTMLButtonElement).style.transform = 'translateY(-1px)'; (e.target as HTMLButtonElement).style.boxShadow = '0 6px 24px color-mix(in srgb, var(--orange) 40%, transparent)'; }}
              onMouseOut={e => { (e.target as HTMLButtonElement).style.transform = 'none'; (e.target as HTMLButtonElement).style.boxShadow = '0 4px 16px color-mix(in srgb, var(--orange) 30%, transparent)'; }}
            >
              Process {files.filter(f => !f.done && !f.error && !f.callId).length > 1
                ? `${files.filter(f => !f.done && !f.error && !f.callId).length} files`
                : files.find(f => !f.done && !f.error && !f.callId)?.name || 'file'}
              {(startTime || endTime) ? ` (${formatSeconds(parseTimeInput(startTime)) || '0:00'} → ${formatSeconds(parseTimeInput(endTime)) || 'end'})` : ''}
            </button>
          </div>
        </div>
      )}

      {/* Upload Queue — files being processed or completed */}
      <div className="upload-queue">
        {files.filter(f => f.callId || f.done || f.error).map((file, i) => (
          <div className="upload-item" key={i}>
            <div className="upload-item-info">
              <div className="upload-item-name" style={{ fontFamily: 'var(--font-mono)' }}>{file.name}</div>
              <div className="upload-progress">
                <div className="upload-progress-fill" style={{ width: `${file.progress}%` }} />
              </div>
            </div>
            <span
              className="upload-item-status"
              style={{
                fontFamily: 'var(--font-mono)',
                color: file.error ? 'var(--danger)' : file.done ? 'var(--success)' : 'var(--text-dim)',
              }}
            >
              {file.error ? `Error: ${file.error.slice(0, 80)}` : file.done ? `Complete${file.callId ? ` (${file.callId})` : ''}` : `Processing... ${file.progress}%`}
            </span>
          </div>
        ))}
      </div>

      {/* Processing View — Live Dashboard */}
      {processing && (
        <div style={{ marginTop: 'var(--sp-6)', display: 'flex', flexDirection: 'column', gap: 'var(--sp-4)' }}>
          {/* Header with elapsed time */}
          <div className="panel">
            <div className="panel-header">
              <h3 style={{ fontFamily: 'var(--font-mono)' }}>Processing Pipeline</h3>
              <div style={{ display: 'flex', gap: 'var(--sp-3)', alignItems: 'center' }}>
                {liveData.elapsed != null && (
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                    {formatElapsed(liveData.elapsed * 1000)}
                  </span>
                )}
                <span className={`badge ${stageStates.every(s => s === 'done') ? 'badge-success' : 'badge-warning'}`} style={{ fontFamily: 'var(--font-mono)' }}>
                  {stageStates.every(s => s === 'done') ? 'Complete' : currentStageName || 'In Progress'}
                </span>
              </div>
            </div>

            {/* Waveform for the file being processed */}
            {files.find(f => f.callId && !f.done) && (
              <div style={{ padding: '0 var(--sp-4) var(--sp-2)' }}>
                <Waveform
                  file={files.find(f => f.callId && !f.done)!.file}
                  trimStart={parseTimeInput(startTime)}
                  trimEnd={parseTimeInput(endTime)}
                />
              </div>
            )}

            {/* Stage pipeline with timing */}
            <div className="processing-stages">
              {stages.map((s, i) => {
                const state = stageStates[i] || 'idle';
                // Find matching stage time
                const stageTimeKeys = Object.keys(liveData.stage_times || {});
                const matchingKey = stageTimeKeys.find(k => {
                  if (i === 0) return k.includes('Normalize') || k.includes('Quality') || k.includes('Cleanup');
                  if (i === 1) return k.includes('WhisperX');
                  if (i === 2) return k.includes('Analysis') && !k.includes('LLM');
                  if (i === 3) return k.includes('LLM');
                  if (i === 4) return k.includes('Output');
                  return false;
                });
                const stageTime = matchingKey ? liveData.stage_times![matchingKey] : null;

                return (
                  <div
                    key={s.id}
                    className={`processing-stage${state === 'active' ? ' active' : ''}${state === 'done' ? ' completed' : ''}`}
                    style={{ '--stage-color': stageColors[i] } as React.CSSProperties}
                  >
                    <div className="stage-indicator" style={{ fontFamily: 'var(--font-mono)' }}>
                      {state === 'done' ? '\u2713' : state === 'active' ? (
                        <span style={{ display: 'inline-block', animation: 'spin 1s linear infinite' }}>&#9881;</span>
                      ) : String(s.id).padStart(2, '0')}
                    </div>
                    <div className="stage-info">
                      <h4 style={{ fontFamily: 'var(--font-mono)' }}>{s.name}</h4>
                      <p>{s.desc}</p>
                    </div>
                    <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
                      {stageTime != null && state === 'done' && (
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                          {stageTime.toFixed(1)}s
                        </span>
                      )}
                      {state === 'active' && (
                        <span className="badge badge-info" style={{ fontFamily: 'var(--font-mono)' }}>Processing</span>
                      )}
                      {state === 'done' && (
                        <span style={{ color: 'var(--success)', fontSize: '1.1rem' }}>{'\u2713'}</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Live metrics cards — appear as stages complete */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 'var(--sp-3)' }}>
            {/* Audio Quality — appears after Stage 2 */}
            {liveData.audio_quality_score != null && (
              <div className="panel" style={{ padding: 'var(--sp-3)' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', marginBottom: 'var(--sp-2)' }}>AUDIO QUALITY</div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--sp-2)' }}>
                  <span style={{
                    fontFamily: 'var(--font-mono)', fontSize: '2rem', fontWeight: 700,
                    color: liveData.audio_quality_score >= 70 ? 'var(--success)' : liveData.audio_quality_score >= 40 ? 'var(--warning)' : 'var(--danger)',
                  }}>
                    {liveData.audio_quality_score}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-dim)' }}>/100</span>
                  <span className={`badge ${liveData.audio_quality_score >= 70 ? 'badge-success' : liveData.audio_quality_score >= 40 ? 'badge-warning' : 'badge-danger'}`}
                    style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', marginLeft: 'auto' }}>
                    {liveData.audio_quality_flag}
                  </span>
                </div>
                {/* Component breakdown bars */}
                {liveData.audio_quality_components && Object.keys(liveData.audio_quality_components).length > 0 && (
                  <div style={{ marginTop: 'var(--sp-2)', display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {(['snr', 'clipping', 'speech_ratio', 'spectral'] as const).map(k => {
                      const val = liveData.audio_quality_components?.[k] ?? 0;
                      const labels: Record<string, string> = { snr: 'SNR', clipping: 'Clipping', speech_ratio: 'Speech', spectral: 'Spectral' };
                      const weights: Record<string, string> = { snr: '40%', clipping: '20%', speech_ratio: '20%', spectral: '20%' };
                      return (
                        <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-dim)', width: 55, textAlign: 'right' }}>
                            {labels[k]} <span style={{ opacity: 0.5 }}>{weights[k]}</span>
                          </span>
                          <div style={{ flex: 1, height: 6, background: 'var(--bg)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{
                              width: `${val}%`, height: '100%', borderRadius: 3,
                              background: val >= 70 ? 'var(--success)' : val >= 40 ? 'var(--warning)' : 'var(--danger)',
                            }} />
                          </div>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-dim)', width: 24 }}>
                            {Math.round(val)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
                <div style={{ marginTop: 'var(--sp-2)', display: 'flex', gap: 'var(--sp-3)', fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-dim)' }}>
                  {liveData.audio_snr_db != null && <span>SNR: {liveData.audio_snr_db}dB</span>}
                  {liveData.audio_speech_pct != null && <span>Speech: {liveData.audio_speech_pct}%</span>}
                  {liveData.audio_duration != null && <span>{formatSeconds(liveData.audio_duration)}</span>}
                </div>
              </div>
            )}

            {/* Transcript info — appears after Stage 3 */}
            {liveData.transcript_segments != null && (
              <div className="panel" style={{ padding: 'var(--sp-3)' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', marginBottom: 'var(--sp-2)' }}>TRANSCRIPTION</div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--sp-2)' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '2rem', fontWeight: 700, color: 'var(--text)' }}>
                    {liveData.transcript_segments}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-dim)' }}>segments</span>
                </div>
                {liveData.transcript_language && (
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: 'var(--sp-1)' }}>
                    Language: <span style={{ color: 'var(--text)' }}>{liveData.transcript_language.toUpperCase()}</span>
                  </div>
                )}
              </div>
            )}

            {/* Stage timing breakdown — appears when we have timing data */}
            {liveData.stage_times && Object.keys(liveData.stage_times).length > 0 && (
              <div className="panel" style={{ padding: 'var(--sp-3)' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', marginBottom: 'var(--sp-2)' }}>TIMING</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {Object.entries(liveData.stage_times).map(([name, secs]) => (
                    <div key={name} style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: '0.65rem' }}>
                      <span style={{ color: 'var(--text-dim)' }}>{name.replace('Stage ', '')}</span>
                      <span style={{ color: secs > 60 ? 'var(--warning)' : 'var(--text)' }}>{secs.toFixed(1)}s</span>
                    </div>
                  ))}
                  {liveData.elapsed != null && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: '0.7rem', borderTop: '1px solid var(--border)', paddingTop: 3, marginTop: 2 }}>
                      <span style={{ color: 'var(--text)' }}>Total elapsed</span>
                      <span style={{ color: 'var(--orange)' }}>{formatElapsed(liveData.elapsed * 1000)}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
