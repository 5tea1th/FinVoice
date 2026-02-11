'use client';

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts';
import { CHART_COLORS } from '@/components/charts/colors';
import ChartTooltip from '@/components/charts/ChartTooltip';
import type { Entity, Obligation } from '@/lib/api';

interface EntitiesTabProps {
  entities: Entity[];
  obligations: Obligation[];
}

const ENTITY_COLORS: Record<string, string> = {
  payment_amount: '#22c55e', currency_amount: '#22c55e', emi_amount: '#22c55e', loan_amount: '#22c55e',
  interest_rate: '#f59e0b', tenure: '#f59e0b',
  due_date: '#3b82f6', date: '#3b82f6',
  organization: '#a855f7', person_name: '#a855f7',
  account_number: '#06b6d4', reference_number: '#06b6d4', pan_number: '#06b6d4', ifsc_code: '#06b6d4',
  product_name: '#ec4899', penalty: '#ef4444',
};

const ENTITY_LABELS: Record<string, string> = {
  payment_amount: 'Amounts', currency_amount: 'Amounts', emi_amount: 'Amounts', loan_amount: 'Amounts',
  interest_rate: 'Rates & Terms', tenure: 'Rates & Terms',
  due_date: 'Dates', date: 'Dates',
  organization: 'People & Orgs', person_name: 'People & Orgs',
  account_number: 'Identifiers', reference_number: 'Identifiers', pan_number: 'Identifiers', ifsc_code: 'Identifiers',
  product_name: 'Products', penalty: 'Penalties',
};

const GROUP_COLORS: Record<string, string> = {
  'Amounts': '#22c55e', 'Rates & Terms': '#f59e0b', 'Dates': '#3b82f6',
  'People & Orgs': '#a855f7', 'Identifiers': '#06b6d4', 'Products': '#ec4899',
  'Penalties': '#ef4444', 'Other': '#94a3b8',
};

export default function EntitiesTab({ entities, obligations }: EntitiesTabProps) {
  if (entities.length === 0 && obligations.length === 0) {
    return (
      <p style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', padding: 'var(--sp-6)', textAlign: 'center' }}>
        No financial entities extracted from this call.
      </p>
    );
  }

  // Group entities
  const groups: Record<string, Entity[]> = {};
  entities.forEach(e => {
    const group = ENTITY_LABELS[e.type] || 'Other';
    if (!groups[group]) groups[group] = [];
    groups[group].push(e);
  });
  const groupOrder = ['Amounts', 'Rates & Terms', 'Dates', 'People & Orgs', 'Identifiers', 'Products', 'Penalties', 'Other'];
  const activeGroups = groupOrder.filter(g => groups[g]);

  // Chart data
  const chartData = activeGroups.map(g => ({ name: g, count: groups[g].length }));

  return (
    <div>
      {/* Entity type distribution chart */}
      {chartData.length > 1 && (
        <div style={{ marginBottom: 'var(--sp-3)' }}>
          <div style={{ width: '100%', height: Math.max(100, chartData.length * 28) }}>
            <ResponsiveContainer>
              <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 80 }}>
                <XAxis type="number" tick={{ fill: CHART_COLORS.textDim, fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={{ stroke: CHART_COLORS.border }} tickLine={false} allowDecimals={false} />
                <YAxis type="category" dataKey="name" tick={{ fill: CHART_COLORS.textDim, fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} width={80} />
                <Tooltip content={<ChartTooltip formatter={(v) => `${v} entities`} />} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={20}>
                  {chartData.map((entry) => (
                    <Cell key={entry.name} fill={GROUP_COLORS[entry.name] || '#94a3b8'} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Entity groups */}
      {activeGroups.map(group => (
        <div key={group} style={{ marginBottom: 'var(--sp-3)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-dim)', marginBottom: 'var(--sp-1)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {group} ({groups[group].length})
          </h4>
          <div className="entity-list">
            {groups[group].map((e, i) => {
              const color = ENTITY_COLORS[e.type] || '#94a3b8';
              return (
                <div className="entity-tag" key={i} style={{ borderLeft: `3px solid ${color}` }}>
                  <span className="entity-type" style={{ fontFamily: 'var(--font-mono)', color }}>{e.type.replace(/_/g, ' ')}</span>
                  <span className="entity-value" style={{ fontFamily: 'var(--font-mono)' }}>{e.value}</span>
                  {e.context && e.context !== e.value && (
                    <span className="entity-context" style={{ fontStyle: 'italic' }}>&ldquo;{e.context}&rdquo;</span>
                  )}
                  <span className="entity-context">{(e.confidence * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        </div>
      ))}

      {obligations.length > 0 && (
        <div style={{ marginTop: 'var(--sp-4)', borderTop: '1px solid var(--border)', paddingTop: 'var(--sp-3)' }}>
          <h4 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--sp-2)', fontSize: '0.85rem' }}>
            Obligations ({obligations.length})
          </h4>
          {obligations.map((o, i) => {
            const strengthBadge = o.strength === 'binding' ? 'badge-danger' : o.strength === 'promise' ? 'badge-warning' : o.strength === 'conditional' ? 'badge-info' : 'badge-success';
            return (
              <div key={i} style={{ padding: 'var(--sp-2)', marginBottom: 'var(--sp-1)', background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 'var(--sp-2)', flexWrap: 'wrap' }}>
                <span className={`badge ${strengthBadge}`} style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>{o.strength}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>[{o.speaker}]</span>
                <span style={{ fontSize: '0.8rem', flex: 1 }}>&ldquo;{o.text}&rdquo;</span>
                {o.legallySignificant && <span className="badge badge-danger" style={{ fontSize: '0.6rem' }}>LEGAL</span>}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
