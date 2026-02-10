'use client';

import { useState, useEffect } from 'react';
import Sidebar, { type View } from '@/components/dashboard/Sidebar';
import Overview, { CallsTable } from '@/components/dashboard/Overview';
import UploadView from '@/components/dashboard/UploadView';
import CallDetail from '@/components/dashboard/CallDetail';
import ReviewQueue from '@/components/dashboard/ReviewQueue';
import SettingsView from '@/components/dashboard/SettingsView';
import { getCalls, type Call } from '@/lib/api';

const VIEW_TITLES: Record<View, string> = {
  overview: 'Overview',
  upload: 'Upload',
  calls: 'Calls',
  review: 'Review Queue',
  settings: 'Settings',
};

export default function DashboardPage() {
  const [activeView, setActiveView] = useState<View>('overview');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedCallId, setSelectedCallId] = useState<string | null>(null);
  const [allCalls, setAllCalls] = useState<Call[]>([]);

  useEffect(() => {
    getCalls().then(setAllCalls);
  }, []);

  const handleNavigate = (view: View) => {
    setActiveView(view);
    setSelectedCallId(null);
    setSidebarOpen(false);
  };

  const handleCallClick = (callId: string) => {
    setSelectedCallId(callId);
    setActiveView('calls');
  };

  return (
    <div className="dashboard">
      <Sidebar activeView={activeView} onNavigate={handleNavigate} open={sidebarOpen} />

      <main className="main-content">
        {/* Header */}
        <header className="main-header">
          <div className="header-left">
            <button
              className="mobile-menu-btn"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              &#x2630;
            </button>
            <h2 className="header-title" style={{ fontFamily: 'var(--font-mono)' }}>
              {VIEW_TITLES[activeView]}
            </h2>
          </div>
          <div className="header-right">
            <div className="search-box">
              <span className="search-icon">&#x1F50D;</span>
              <input
                type="text"
                placeholder="Search calls..."
                className="search-input"
                style={{ fontFamily: 'var(--font-mono)' }}
              />
            </div>
            <button className="notification-btn">
              <span>&#x1F514;</span>
              <span className="notif-badge" style={{ fontFamily: 'var(--font-mono)' }}>3</span>
            </button>
          </div>
        </header>

        {/* Content */}
        <div className="content-area">
          {activeView === 'overview' && (
            <div style={{ animation: 'view-in .3s ease' }}>
              <Overview onCallClick={handleCallClick} />
            </div>
          )}

          {activeView === 'upload' && (
            <div style={{ animation: 'view-in .3s ease' }}>
              <UploadView />
            </div>
          )}

          {activeView === 'calls' && (
            <div style={{ animation: 'view-in .3s ease' }}>
              {selectedCallId ? (
                <CallDetail callId={selectedCallId} onBack={() => setSelectedCallId(null)} />
              ) : (
                <div className="panel">
                  <div className="panel-header">
                    <h3 style={{ fontFamily: 'var(--font-mono)' }}>All Processed Calls</h3>
                  </div>
                  <CallsTable calls={allCalls} onCallClick={handleCallClick} showAgent />
                </div>
              )}
            </div>
          )}

          {activeView === 'review' && (
            <div style={{ animation: 'view-in .3s ease' }}>
              <ReviewQueue />
            </div>
          )}

          {activeView === 'settings' && (
            <div style={{ animation: 'view-in .3s ease' }}>
              <SettingsView />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
