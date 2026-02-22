'use client';

/**
 * Production Readiness Checklist for StudyBuddy v14
 *
 * Fetches and displays the production checklist status across
 * security, performance, monitoring, database, and external service categories.
 */

import { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '../lib/api';

interface CheckItem {
  name: string;
  status: 'green' | 'yellow' | 'red';
  message: string;
  detail?: string;
}

interface ChecklistResponse {
  overall_status: 'green' | 'yellow' | 'red';
  categories: Record<string, CheckItem[]>;
  check_count: number;
  timestamp: string;
}

const CATEGORY_LABELS: Record<string, string> = {
  security: 'Security',
  performance: 'Performance',
  monitoring: 'Monitoring',
  database: 'Database',
  external: 'External Services',
};

const STATUS_CONFIG: Record<
  'green' | 'yellow' | 'red',
  { label: string; borderColor: string; backgroundColor: string; color: string }
> = {
  green: {
    label: 'Production Ready',
    borderColor: 'var(--color-success)',
    backgroundColor: 'hsl(142 71% 45% / 0.1)',
    color: 'var(--color-success)',
  },
  yellow: {
    label: 'Needs Attention',
    borderColor: 'var(--color-warning)',
    backgroundColor: 'hsl(38 92% 50% / 0.1)',
    color: 'var(--color-warning)',
  },
  red: {
    label: 'Not Ready',
    borderColor: 'var(--color-error)',
    backgroundColor: 'hsl(0 84% 60% / 0.1)',
    color: 'var(--color-error)',
  },
};

function StatusIcon({ status }: { status: 'green' | 'yellow' | 'red' }) {
  if (status === 'green') {
    return (
      <span
        style={{ color: 'var(--color-success)', fontSize: '1.1rem', flexShrink: 0 }}
        aria-label="Passing"
      >
        &#x2705;
      </span>
    );
  }
  if (status === 'yellow') {
    return (
      <span
        style={{ color: 'var(--color-warning)', fontSize: '1.1rem', flexShrink: 0 }}
        aria-label="Warning"
      >
        &#x26A0;&#xFE0F;
      </span>
    );
  }
  return (
    <span
      style={{ color: 'var(--color-error)', fontSize: '1.1rem', flexShrink: 0 }}
      aria-label="Failing"
    >
      &#x274C;
    </span>
  );
}

export default function ProductionChecklist() {
  const [data, setData] = useState<ChecklistResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchChecklist = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await apiFetch('/api/production-checklist');
      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }
      const json: ChecklistResponse = await res.json();
      setData(json);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to load production checklist'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchChecklist();
  }, [fetchChecklist]);

  if (loading && !data) {
    return (
      <div className="content-panel" style={{ padding: 'var(--spacing-xl)' }}>
        <h2 style={{ marginBottom: 'var(--spacing-md)' }}>Production Readiness</h2>
        <div className="loading">
          <div className="spinner" />
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="content-panel" style={{ padding: 'var(--spacing-xl)' }}>
        <h2 style={{ marginBottom: 'var(--spacing-md)' }}>Production Readiness</h2>
        <div
          className="alert"
          style={{
            padding: 'var(--spacing-sm) var(--spacing-md)',
            backgroundColor: 'hsl(0 84% 60% / 0.1)',
            border: '1px solid hsl(0 84% 60% / 0.3)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--color-error)',
            fontSize: 'var(--font-sm)',
          }}
        >
          {error}
        </div>
      </div>
    );
  }

  if (!data) return null;

  const overallConfig = STATUS_CONFIG[data.overall_status];
  const categoryOrder = ['security', 'performance', 'monitoring', 'database', 'external'];

  return (
    <div className="content-panel" style={{ padding: 'var(--spacing-xl)' }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 'var(--spacing-lg)',
        }}
      >
        <h2 style={{ margin: 0 }}>Production Readiness</h2>
        <button
          className="btn-secondary btn-sm"
          onClick={fetchChecklist}
          disabled={loading}
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* Overall status banner */}
      <div
        style={{
          padding: 'var(--spacing-md) var(--spacing-lg)',
          borderRadius: 'var(--radius-lg)',
          border: `2px solid ${overallConfig.borderColor}`,
          backgroundColor: overallConfig.backgroundColor,
          marginBottom: 'var(--spacing-xl)',
          textAlign: 'center',
        }}
      >
        <span
          style={{
            fontSize: 'var(--font-xl)',
            fontWeight: 600,
            color: overallConfig.color,
          }}
        >
          {overallConfig.label}
        </span>
        <span
          style={{
            marginLeft: 'var(--spacing-md)',
            fontSize: 'var(--font-sm)',
            color: 'var(--color-text-secondary)',
          }}
        >
          {data.check_count} checks evaluated
        </span>
      </div>

      {/* Error banner (for refresh errors when we already have data) */}
      {error && (
        <div
          className="alert"
          style={{
            padding: 'var(--spacing-sm) var(--spacing-md)',
            backgroundColor: 'hsl(0 84% 60% / 0.1)',
            border: '1px solid hsl(0 84% 60% / 0.3)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--color-error)',
            fontSize: 'var(--font-sm)',
            marginBottom: 'var(--spacing-lg)',
          }}
        >
          Refresh failed: {error}
        </div>
      )}

      {/* Categories */}
      {categoryOrder.map((categoryKey) => {
        const checks = data.categories[categoryKey];
        if (!checks || checks.length === 0) return null;

        const label = CATEGORY_LABELS[categoryKey] || categoryKey;

        return (
          <div key={categoryKey} style={{ marginBottom: 'var(--spacing-xl)' }}>
            <h3
              style={{
                fontSize: 'var(--font-lg)',
                fontWeight: 600,
                marginBottom: 'var(--spacing-md)',
                paddingBottom: 'var(--spacing-xs)',
                borderBottom: '1px solid var(--color-border)',
              }}
            >
              {label}
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
              {checks.map((check, idx) => (
                <div
                  key={`${categoryKey}-${idx}`}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 'var(--spacing-sm)',
                    padding: 'var(--spacing-sm) var(--spacing-md)',
                    backgroundColor: 'var(--color-bg-card)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-md)',
                  }}
                >
                  <StatusIcon status={check.status} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 600, fontSize: 'var(--font-sm)' }}>
                      {check.name}
                    </div>
                    <div
                      style={{
                        fontSize: 'var(--font-sm)',
                        color: 'var(--color-text-secondary)',
                        marginTop: '2px',
                      }}
                    >
                      {check.message}
                    </div>
                    {check.detail && (
                      <div
                        style={{
                          fontSize: 'var(--font-xs)',
                          color: 'var(--color-text-muted)',
                          marginTop: '4px',
                        }}
                      >
                        {check.detail}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {/* Timestamp footer */}
      <div
        style={{
          textAlign: 'center',
          fontSize: 'var(--font-xs)',
          color: 'var(--color-text-muted)',
          marginTop: 'var(--spacing-lg)',
          paddingTop: 'var(--spacing-md)',
          borderTop: '1px solid var(--color-border)',
        }}
      >
        Last checked: {new Date(data.timestamp).toLocaleString()}
      </div>
    </div>
  );
}
