'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { getBackboardStatus, type BackboardStatus } from '@/lib/api';

type View = 'overview' | 'upload' | 'calls' | 'review' | 'exports' | 'settings';

interface SidebarProps {
  activeView: View;
  onNavigate: (view: View) => void;
  open: boolean;
}

const NAV_ITEMS: { view: View; icon: string; label: string }[] = [
  { view: 'overview', icon: '\u25A3', label: 'Overview' },
  { view: 'upload', icon: '\u21E7', label: 'Upload' },
  { view: 'calls', icon: '\u260E', label: 'Calls' },
  { view: 'review', icon: '\u2691', label: 'Review Queue' },
  { view: 'exports', icon: '\u2913', label: 'Exports' },
  { view: 'settings', icon: '\u2699', label: 'Settings' },
];

export default function Sidebar({ activeView, onNavigate, open }: SidebarProps) {
  const [bbStatus, setBbStatus] = useState<BackboardStatus | null>(null);

  useEffect(() => {
    getBackboardStatus().then(setBbStatus);
  }, []);

  return (
    <aside className={`sidebar${open ? ' open' : ''}`}>
      <div className="sidebar-header">
        <Link href="/" className="sidebar-logo" style={{ fontFamily: 'var(--font-mono)' }}>
          Fin<span>Voice</span>
        </Link>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(item => (
          <button
            key={item.view}
            className={`sidebar-item${activeView === item.view ? ' active' : ''}`}
            onClick={() => onNavigate(item.view)}
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            <span className="sidebar-icon">{item.icon}</span>
            <span className="sidebar-label">{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-status" style={{ marginBottom: 'var(--sp-2)' }}>
          <span className="status-dot" />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Pipeline active</span>
        </div>
        {bbStatus && (
          <div className="sidebar-status">
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: bbStatus.configured ? 'var(--s5)' : 'var(--text-dim)',
              display: 'inline-block', flexShrink: 0,
              animation: bbStatus.configured ? 'pulse 2s ease-in-out infinite' : 'none',
            }} />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
              {bbStatus.configured
                ? `AI Memory (${bbStatus.calls_stored || 0})`
                : 'AI Memory off'}
            </span>
          </div>
        )}
      </div>
    </aside>
  );
}

export type { View };
