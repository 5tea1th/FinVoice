'use client';

import { useEffect, useState } from 'react';
import { getStats, getCalls, type Call, type Stats } from '@/lib/api';

interface OverviewProps {
  onCallClick: (callId: string) => void;
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

  const statItems = stats ? [
    { label: 'Total Calls Processed', value: stats.totalCalls.toLocaleString(), color: 'var(--s1)', bars: [40,65,50,80,70,55,90,60,75,85] },
    { label: 'Avg Compliance Score', value: stats.avgCompliance + '%', color: 'var(--s2)', bars: [70,75,65,80,85,78,82,88,79,81] },
    { label: 'Fraud Alerts', value: String(stats.fraudAlerts), color: 'var(--s4)', bars: [20,35,15,45,30,25,40,50,35,23] },
    { label: 'Pending Reviews', value: String(stats.pendingReviews), color: 'var(--s3)', bars: [50,40,60,35,55,45,30,65,50,18] },
  ] : [];

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 'var(--sp-12)', color: 'var(--text-dim)' }}>
        <span style={{ fontFamily: 'var(--font-mono)' }}>Loading dashboard...</span>
      </div>
    );
  }

  return (
    <>
      {/* Stats */}
      <div className="stats-row">
        {statItems.map(item => (
          <div className="stat-card" key={item.label} style={{ '--stat-color': item.color } as React.CSSProperties}>
            <div className="stat-label" style={{ fontFamily: 'var(--font-mono)' }}>{item.label}</div>
            <div className="stat-value" style={{ fontFamily: 'var(--font-mono)' }}>{item.value}</div>
            <div className="stat-sparkline">
              {item.bars.map((h, i) => (
                <div className="spark-bar" key={i} style={{ height: `${h}%`, background: item.color }} />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Pipeline Status */}
      <div className="panel">
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
                className={`pipeline-stage-bar${i === 2 ? ' active-stage' : ''}`}
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

      {/* Recent Activity */}
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

export function CallsTable({ calls, onCallClick, showAgent }: { calls: Call[]; onCallClick: (id: string) => void; showAgent?: boolean }) {
  const compColor = (score: number) => score >= 80 ? 'var(--success)' : score >= 60 ? 'var(--warning)' : 'var(--danger)';
  const riskBadge = (risk: string) => risk === 'high' ? 'badge-danger' : risk === 'medium' ? 'badge-warning' : 'badge-success';
  const statusBadge = (s: string) => s === 'escalated' ? 'badge-danger' : s === 'flagged' ? 'badge-warning' : s === 'pending' ? 'badge-info' : 'badge-success';

  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th style={{ fontFamily: 'var(--font-mono)' }}>Call ID</th>
            <th style={{ fontFamily: 'var(--font-mono)' }}>Date</th>
            <th style={{ fontFamily: 'var(--font-mono)' }}>Duration</th>
            <th style={{ fontFamily: 'var(--font-mono)' }}>Language</th>
            {showAgent && <th style={{ fontFamily: 'var(--font-mono)' }}>Agent</th>}
            <th style={{ fontFamily: 'var(--font-mono)' }}>Compliance</th>
            <th style={{ fontFamily: 'var(--font-mono)' }}>Fraud Risk</th>
            <th style={{ fontFamily: 'var(--font-mono)' }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {calls.map(call => (
            <tr key={call.id} onClick={() => onCallClick(call.id)}>
              <td className="call-id" style={{ fontFamily: 'var(--font-mono)' }}>{call.id}</td>
              <td>{call.date}</td>
              <td>{call.duration}</td>
              <td>{call.language}</td>
              {showAgent && <td>{call.agent}</td>}
              <td>
                <div className="compliance-bar">
                  <div className="compliance-track">
                    <div className="compliance-fill" style={{ width: `${call.complianceScore}%`, background: compColor(call.complianceScore) }} />
                  </div>
                  <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>{call.complianceScore}%</span>
                </div>
              </td>
              <td><span className={`badge ${riskBadge(call.fraudRisk)}`} style={{ fontFamily: 'var(--font-mono)' }}>{call.fraudRisk}</span></td>
              <td><span className={`badge ${statusBadge(call.status)}`} style={{ fontFamily: 'var(--font-mono)' }}>{call.status}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
