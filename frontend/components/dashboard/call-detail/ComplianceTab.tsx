'use client';

import { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { CHART_COLORS } from '@/components/charts/colors';
import { complianceReason, type ComplianceRule } from '@/lib/api';

interface ComplianceTabProps {
  compliance: ComplianceRule[];
}

export default function ComplianceTab({ compliance }: ComplianceTabProps) {
  const [reasoning, setReasoning] = useState<Record<number, { loading: boolean; text: string }>>({});

  if (compliance.length === 0) {
    return (
      <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.875rem', padding: 'var(--sp-4)' }}>
        No compliance checks available.
      </p>
    );
  }

  const passed = compliance.filter(c => c.passed).length;
  const failed = compliance.length - passed;
  const donutData = [
    { name: 'Passed', value: passed },
    { name: 'Failed', value: failed },
  ].filter(d => d.value > 0);

  const handleCloudAnalysis = async (index: number, rule: ComplianceRule) => {
    setReasoning(prev => ({ ...prev, [index]: { loading: true, text: '' } }));
    const result = await complianceReason(
      rule.detail,
      `Compliance check: ${rule.rule}`,
      'RBI Fair Practice Code / SEBI KYC Guidelines'
    );
    setReasoning(prev => ({
      ...prev,
      [index]: { loading: false, text: result.reasoning || result.error || 'No response' },
    }));
  };

  return (
    <div>
      {/* Donut + Stats */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-4)', marginBottom: 'var(--sp-4)', padding: '0 var(--sp-2)' }}>
        <div style={{ width: 80, height: 80 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie data={donutData} innerRadius={22} outerRadius={35} paddingAngle={4} dataKey="value" stroke="none">
                {donutData.map((entry) => (
                  <Cell key={entry.name} fill={entry.name === 'Passed' ? CHART_COLORS.success : CHART_COLORS.danger} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: CHART_COLORS.success, fontWeight: 600 }}>
            {passed} passed
          </span>
          {failed > 0 && (
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: CHART_COLORS.danger, fontWeight: 600, marginLeft: 'var(--sp-3)' }}>
              {failed} failed
            </span>
          )}
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: 2 }}>
            Score: {Math.round((passed / compliance.length) * 100)}%
          </div>
        </div>
      </div>

      {/* Compliance checks */}
      <div className="compliance-list">
        {compliance.map((c, i) => (
          <div key={i}>
            <div className="compliance-rule" style={{ borderLeft: `3px solid ${c.passed ? 'var(--success)' : 'var(--danger)'}` }}>
              <span className={`compliance-icon ${c.passed ? 'pass' : 'fail'}`}>
                {c.passed ? '\u2713' : '\u2717'}
              </span>
              <div style={{ flex: 1 }}>
                <h4 style={{ fontFamily: 'var(--font-mono)' }}>{c.rule}</h4>
                <p>{c.detail}</p>
              </div>
              {!c.passed && (
                <button
                  onClick={() => handleCloudAnalysis(i, c)}
                  disabled={reasoning[i]?.loading}
                  style={{
                    fontFamily: 'var(--font-mono)', fontSize: '0.65rem', padding: '3px 8px',
                    borderRadius: 6, border: `1px solid ${CHART_COLORS.s5}`, color: CHART_COLORS.s5,
                    background: 'transparent', cursor: 'pointer', whiteSpace: 'nowrap',
                    opacity: reasoning[i]?.loading ? 0.5 : 1,
                  }}
                >
                  {reasoning[i]?.loading ? 'Analyzing...' : 'Cloud Analysis'}
                </button>
              )}
            </div>
            {reasoning[i]?.text && (
              <div style={{
                margin: '-4px var(--sp-3) var(--sp-3)', padding: 'var(--sp-3)',
                background: 'rgba(168,85,247,0.06)', border: `1px solid ${CHART_COLORS.s5}33`,
                borderRadius: 'var(--radius-sm)', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.6,
              }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: CHART_COLORS.s5, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  GPT-4o Analysis
                </div>
                {reasoning[i].text}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
