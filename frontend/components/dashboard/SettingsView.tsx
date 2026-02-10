'use client';

export default function SettingsView() {
  return (
    <div className="panel">
      <div className="panel-header">
        <h3 style={{ fontFamily: 'var(--font-mono)' }}>Settings</h3>
      </div>
      <div className="settings-grid">
        <div className="setting-item">
          <div>
            <h4 style={{ fontFamily: 'var(--font-mono)' }}>Pipeline Auto-Processing</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-dim)' }}>Automatically process uploaded files through the full pipeline</p>
          </div>
          <label className="toggle">
            <input type="checkbox" defaultChecked />
            <span className="toggle-slider" />
          </label>
        </div>
        <div className="setting-item">
          <div>
            <h4 style={{ fontFamily: 'var(--font-mono)' }}>Compliance Alerts</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-dim)' }}>Send notifications for compliance scores below threshold</p>
          </div>
          <label className="toggle">
            <input type="checkbox" defaultChecked />
            <span className="toggle-slider" />
          </label>
        </div>
        <div className="setting-item">
          <div>
            <h4 style={{ fontFamily: 'var(--font-mono)' }}>Fraud Alert Threshold</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-dim)' }}>Risk level that triggers automatic escalation</p>
          </div>
          <select className="filter-select" style={{ fontFamily: 'var(--font-mono)' }}>
            <option>Medium</option>
            <option>High</option>
            <option>Critical Only</option>
          </select>
        </div>
        <div className="setting-item">
          <div>
            <h4 style={{ fontFamily: 'var(--font-mono)' }}>Export Format</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-dim)' }}>Default format for ML dataset export</p>
          </div>
          <select className="filter-select" style={{ fontFamily: 'var(--font-mono)' }}>
            <option>JSON</option>
            <option>CSV</option>
            <option>Parquet</option>
          </select>
        </div>
      </div>
    </div>
  );
}
