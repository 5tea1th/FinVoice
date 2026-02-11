'use client';

import { useEffect, useState } from 'react';
import { getStats, getCalls, type Call, type Stats } from '@/lib/api';
import { RiskDonut, ComplianceDistribution, CHART_COLORS } from '@/components/charts';

interface OverviewProps {
  onCallClick: (callId: string) => void;
}

function formatDuration(totalSec: number): string {
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export default function Overview({ onCallClick }: OverviewProps) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [calls, setCalls] = useState<Call[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getStats().then(setStats),
      getCalls().then(setCalls),
    ]).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 'var(--sp-12)', color: 'var(--text-dim)' }}>
        <span style={{ fontFamily: 'var(--font-mono)' }}>Loading dashboard...</span>
      </div>
    );
  }

  const avgDuration = stats && stats.totalCalls > 0
    ? formatDuration(stats.totalDurationSeconds / stats.totalCalls)
    : '—';

  const langEntries = stats ? Object.entries(stats.languages).sort((a, b) => b[1] - a[1]) : [];
  const langTotal = langEntries.reduce((s, [, v]) => s + v, 0);

  // Risk counts for stat card
  const highRisk = stats ? (stats.riskDistribution.high || 0) + (stats.riskDistribution.critical || 0) : 0;

  return (
    <>
      {/* ── Row 1: Stat Cards ── */}
      <div className="stats-row">
        {[
          { label: 'Total Calls', value: stats?.totalCalls.toLocaleString() || '0', color: 'var(--s1)', sub: formatDuration(stats?.totalDurationSeconds || 0) + ' total' },
          { label: 'Avg Compliance', value: (stats?.avgCompliance || 0) + '%', color: 'var(--s2)', sub: calls.length > 0 ? `${calls.filter(c => c.complianceScore >= 80).length}/${calls.length} above 80%` : '' },
          { label: 'Fraud Alerts', value: String(stats?.fraudAlerts || 0), color: 'var(--s4)', sub: highRisk > 0 ? `${highRisk} high/critical risk` : 'No high risk calls' },
          { label: 'Pending Reviews', value: String(stats?.pendingReviews || 0), color: 'var(--s3)', sub: `avg ${avgDuration} per call` },
        ].map(item => (
          <div className="stat-card" key={item.label} style={{ '--stat-color': item.color } as React.CSSProperties}>
            <div className="stat-label" style={{ fontFamily: 'var(--font-mono)' }}>{item.label}</div>
            <div className="stat-value" style={{ fontFamily: 'var(--font-mono)' }}>{item.value}</div>
            {item.sub && (
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 'var(--sp-2)' }}>
                {item.sub}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* ── Row 2: Risk Donut + Compliance Histogram ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--sp-4)', marginBottom: 'var(--sp-5)' }}>
        {/* Risk Distribution */}
        <div className="chart-panel">
          <div className="chart-title">Risk Distribution</div>
          {stats && Object.keys(stats.riskDistribution).length > 0 ? (
            <>
              <RiskDonut distribution={stats.riskDistribution} />
              <div style={{ display: 'flex', justifyContent: 'center', gap: 'var(--sp-4)', marginTop: 'var(--sp-2)' }}>
                {Object.entries(stats.riskDistribution).map(([level, count]) => {
                  const colors: Record<string, string> = { low: CHART_COLORS.riskLow, medium: CHART_COLORS.riskMedium, high: CHART_COLORS.riskHigh, critical: CHART_COLORS.riskCritical };
                  return (
                    <span key={level} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: colors[level] || '#6b7280', display: 'inline-block' }} />
                      {level}: {count}
                    </span>
                  );
                })}
              </div>
            </>
          ) : (
            <div style={{ padding: 'var(--sp-8)', textAlign: 'center', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
              No risk data available
            </div>
          )}
        </div>

        {/* Compliance Score Distribution */}
        <div className="chart-panel">
          <div className="chart-title">Compliance Score Distribution</div>
          {calls.length > 0 ? (
            <ComplianceDistribution calls={calls} />
          ) : (
            <div style={{ padding: 'var(--sp-8)', textAlign: 'center', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
              No compliance data available
            </div>
          )}
        </div>
      </div>

      {/* ── Row 3: Language Distribution + Duration Stats ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--sp-4)', marginBottom: 'var(--sp-5)' }}>
        {/* Language Distribution */}
        <div className="chart-panel">
          <div className="chart-title">Languages Detected</div>
          {langEntries.length > 0 ? (
            <div>
              {/* Stacked bar */}
              <div style={{ display: 'flex', height: 28, borderRadius: 14, overflow: 'hidden', border: `1px solid ${CHART_COLORS.border}`, marginBottom: 'var(--sp-3)' }}>
                {langEntries.map(([lang, count], i) => {
                  const pct = (count / langTotal) * 100;
                  const langColors = [CHART_COLORS.s1, CHART_COLORS.s2, CHART_COLORS.s3, CHART_COLORS.s5, CHART_COLORS.info];
                  return (
                    <div key={lang} style={{
                      width: `${pct}%`, background: langColors[i % langColors.length],
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }} title={`${lang.toUpperCase()}: ${count} calls (${pct.toFixed(0)}%)`}>
                      {pct > 12 && (
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: '#fff', fontWeight: 600 }}>
                          {lang.toUpperCase()}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 16px' }}>
                {langEntries.map(([lang, count], i) => {
                  const langColors = [CHART_COLORS.s1, CHART_COLORS.s2, CHART_COLORS.s3, CHART_COLORS.s5, CHART_COLORS.info];
                  return (
                    <span key={lang} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: langColors[i % langColors.length] }} />
                      {lang.toUpperCase()}: {count} ({((count / langTotal) * 100).toFixed(0)}%)
                    </span>
                  );
                })}
              </div>
            </div>
          ) : (
            <div style={{ padding: 'var(--sp-6)', textAlign: 'center', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
              No language data
            </div>
          )}
        </div>

        {/* Duration & Pipeline Stats */}
        <div className="chart-panel">
          <div className="chart-title">Portfolio Stats</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--sp-3)' }}>
            {[
              { label: 'Total Duration', value: formatDuration(stats?.totalDurationSeconds || 0), color: CHART_COLORS.s1 },
              { label: 'Avg Duration', value: avgDuration, color: CHART_COLORS.s2 },
              { label: 'Languages', value: String(langEntries.length), color: CHART_COLORS.s3 },
              { label: 'Review Rate', value: stats && stats.totalCalls > 0 ? `${((stats.pendingReviews / stats.totalCalls) * 100).toFixed(0)}%` : '—', color: CHART_COLORS.warning },
            ].map(s => (
              <div key={s.label} style={{
                background: 'rgba(255,255,255,0.02)', border: `1px solid ${CHART_COLORS.border}`,
                borderRadius: 8, padding: 'var(--sp-3)', textAlign: 'center',
              }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.3rem', fontWeight: 700, color: s.color }}>{s.value}</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: CHART_COLORS.textDim, textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 2 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Row 4: Pipeline Status (condensed) ── */}
      <div className="panel" style={{ marginBottom: 'var(--sp-5)' }}>
        <div className="panel-header">
          <h3 style={{ fontFamily: 'var(--font-mono)' }}>Pipeline Status</h3>
          <span className="badge badge-success" style={{ fontFamily: 'var(--font-mono)' }}>Active</span>
        </div>
        <div className="pipeline-bar">
          {['Ingest', 'Transcribe', 'Extract', 'Analyze', 'Store'].map((name, i) => {
            const colors = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)'];
            return (
              <div
                key={name}
                className="pipeline-stage-bar"
                style={{ '--bar-color': colors[i] } as React.CSSProperties}
              >
                <span className="bar-label" style={{ fontFamily: 'var(--font-mono)' }}>{name}</span>
              </div>
            );
          })}
        </div>
        <p style={{ marginTop: 'var(--sp-3)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
          {stats?.totalCalls || 0} calls processed &middot; Pipeline ready
        </p>
      </div>

      {/* ── Row 5: Recent Activity ── */}
      <div className="panel">
        <div className="panel-header">
          <h3 style={{ fontFamily: 'var(--font-mono)' }}>Recent Activity</h3>
          <button className="btn btn-ghost" style={{ fontSize: '0.875rem' }}>View all &rarr;</button>
        </div>
        {calls.length > 0 ? (
          <CallsTable calls={calls} onCallClick={onCallClick} />
        ) : (
          <div style={{ padding: 'var(--sp-8)', textAlign: 'center', color: 'var(--text-dim)' }}>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>No calls processed yet. Upload audio files to get started.</p>
          </div>
        )}
      </div>
    </>
  );
}

export function CallsTable({ calls, onCallClick }: { calls: Call[]; onCallClick: (id: string) => void; showAgent?: boolean }) {
  const compColor = (score: number) => score >= 80 ? 'var(--success)' : score >= 60 ? 'var(--warning)' : 'var(--danger)';
  const riskBadge = (risk: string) => risk === 'high' ? 'badge-danger' : risk === 'medium' ? 'badge-warning' : 'badge-success';

  return (
    <div className="call-card-grid">
      {calls.map(call => (
        <div key={call.id} className="call-card" onClick={() => onCallClick(call.id)}>
          <div className="call-card-top">
            <span className="call-card-id">{call.id}</span>
            <span className={`badge ${riskBadge(call.fraudRisk)}`} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem' }}>
              {call.fraudRisk}
            </span>
          </div>
          {call.summary && (
            <p className="call-card-summary">{call.summary}</p>
          )}
          <div className="call-card-meta">
            <span>{call.date}</span>
            <span>{call.duration}</span>
            <span>{call.language.toUpperCase()}</span>
          </div>
          <div className="call-card-bottom">
            <div className="call-card-compliance">
              <div className="call-card-comp-track">
                <div className="call-card-comp-fill" style={{ width: `${call.complianceScore}%`, background: compColor(call.complianceScore) }} />
              </div>
              <span style={{ color: compColor(call.complianceScore) }}>{call.complianceScore}%</span>
            </div>
            {call.tags && call.tags.length > 0 && (
              <span className="call-card-tag">{call.tags[0]}</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
