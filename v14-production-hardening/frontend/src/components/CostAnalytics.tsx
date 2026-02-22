'use client'

/**
 * Cost Analytics Dashboard for StudyBuddy v14
 *
 * Displays per-feature cost breakdown, daily spend trends,
 * budget status, and budget configuration controls.
 */

import { useState, useEffect } from 'react'
import { apiFetch } from '../lib/api'

/* ------------------------------------------------------------------ */
/*  Type definitions                                                   */
/* ------------------------------------------------------------------ */

interface FeatureCost {
    task_type: string
    total_cost_cents: number
    total_input_tokens: number
    total_output_tokens: number
    call_count: number
}

interface DailyCost {
    date: string
    cost_cents: number
}

interface BudgetStatus {
    spend_cents: number
    budget_cents: number
    pct_used: number
    alert_triggered: boolean
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function CostAnalytics() {
    const [featureCosts, setFeatureCosts] = useState<FeatureCost[]>([])
    const [dailyCosts, setDailyCosts] = useState<DailyCost[]>([])
    const [budgetStatus, setBudgetStatus] = useState<BudgetStatus | null>(null)
    const [budgetCents, setBudgetCents] = useState<number>(0)
    const [warningPct, setWarningPct] = useState<number>(80)
    const [days, setDays] = useState<number>(30)
    const [loading, setLoading] = useState<boolean>(true)
    const [error, setError] = useState<string | null>(null)
    const [saving, setSaving] = useState<boolean>(false)
    const [saveSuccess, setSaveSuccess] = useState<string | null>(null)

    /* ----- data loading ------------------------------------------- */

    useEffect(() => {
        loadData()
    }, [days])

    async function loadData() {
        setLoading(true)
        setError(null)

        try {
            const [featureRes, dailyRes, budgetRes] = await Promise.all([
                apiFetch(`/api/costs/by-feature?days=${days}`),
                apiFetch(`/api/costs/daily?days=${days}`),
                apiFetch('/api/budget/status'),
            ])

            const featureData = await featureRes.json()
            const dailyData = await dailyRes.json()
            const budgetData = await budgetRes.json()

            setFeatureCosts(featureData.costs ?? featureData ?? [])
            setDailyCosts(dailyData.costs ?? dailyData ?? [])
            setBudgetStatus(budgetData)

            // Pre-fill the budget form with current values
            if (budgetData) {
                setBudgetCents(budgetData.budget_cents ?? 0)
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load cost data')
        } finally {
            setLoading(false)
        }
    }

    /* ----- budget save -------------------------------------------- */

    async function handleSaveBudget(e: React.FormEvent) {
        e.preventDefault()
        setSaving(true)
        setError(null)
        setSaveSuccess(null)

        try {
            await apiFetch('/api/budget', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    monthly_budget_cents: budgetCents,
                    warning_threshold_pct: warningPct,
                }),
            })
            setSaveSuccess('Budget configuration saved!')
            setTimeout(() => setSaveSuccess(null), 3000)
            // Reload budget status to reflect changes
            const res = await apiFetch('/api/budget/status')
            setBudgetStatus(await res.json())
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to save budget')
        } finally {
            setSaving(false)
        }
    }

    /* ----- helpers ------------------------------------------------ */

    function centsToUsd(cents: number): string {
        return `$${(cents / 100).toFixed(2)}`
    }

    function formatTokens(n: number): string {
        return n.toLocaleString()
    }

    function formatTaskType(t: string): string {
        return t
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (c) => c.toUpperCase())
    }

    const totalCalls = featureCosts.reduce((sum, f) => sum + f.call_count, 0)
    const maxDailyCost = dailyCosts.length > 0
        ? Math.max(...dailyCosts.map((d) => d.cost_cents))
        : 0

    /* ----- loading / error states --------------------------------- */

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
            </div>
        )
    }

    /* ----- render ------------------------------------------------- */

    return (
        <div>
            {/* Page header */}
            <div className="page-header">
                <div className="page-header-content">
                    <h2>Cost Analytics</h2>
                    <p className="page-description">
                        Track API spend by feature, monitor daily trends, and manage your budget.
                    </p>
                </div>
                <div className="page-header-actions">
                    <select
                        className="cost-period-select"
                        value={days}
                        onChange={(e) => setDays(Number(e.target.value))}
                        title="Time period"
                    >
                        <option value={7}>Last 7 days</option>
                        <option value={30}>Last 30 days</option>
                        <option value={90}>Last 90 days</option>
                        <option value={365}>Last year</option>
                    </select>
                </div>
            </div>

            {/* Global error banner */}
            {error && (
                <div className="card card-error mb-lg" role="alert">
                    <p className="text-error">{error}</p>
                </div>
            )}

            {/* Budget alert warning banner */}
            {budgetStatus?.alert_triggered && (
                <div className="card card-warning mb-lg" role="alert">
                    <p>
                        <strong>Budget Warning:</strong> You have used{' '}
                        {budgetStatus.pct_used.toFixed(1)}% of your monthly budget (
                        {centsToUsd(budgetStatus.spend_cents)} of{' '}
                        {centsToUsd(budgetStatus.budget_cents)}).
                    </p>
                </div>
            )}

            {/* ---- Stats grid ---- */}
            <div className="stats-grid mb-lg">
                <div className="stat-card">
                    <div className="stat-value">
                        {budgetStatus ? centsToUsd(budgetStatus.spend_cents) : '$0.00'}
                    </div>
                    <div className="stat-label">Total Spend</div>
                    <div className="stat-sublabel">Last {days} days</div>
                </div>

                <div className="stat-card">
                    <div className="stat-value">
                        {budgetStatus && budgetStatus.budget_cents > 0
                            ? centsToUsd(budgetStatus.budget_cents)
                            : 'Not set'}
                    </div>
                    <div className="stat-label">Monthly Budget</div>
                    <div className="stat-sublabel">Configured limit</div>
                </div>

                <div className="stat-card">
                    <div className="stat-value">
                        {budgetStatus && budgetStatus.budget_cents > 0
                            ? `${budgetStatus.pct_used.toFixed(1)}%`
                            : 'N/A'}
                    </div>
                    <div className="stat-label">Budget Used</div>
                    <div className="stat-sublabel">Current period</div>
                </div>

                <div className="stat-card">
                    <div className="stat-value">{totalCalls.toLocaleString()}</div>
                    <div className="stat-label">API Calls</div>
                    <div className="stat-sublabel">Last {days} days</div>
                </div>
            </div>

            {/* ---- Cost by Feature table ---- */}
            <div className="card mb-lg">
                <h3 className="mb-md">Cost by Feature</h3>

                {featureCosts.length === 0 ? (
                    <p className="text-muted text-sm">No cost data for this period.</p>
                ) : (
                    <div className="cost-table-wrap">
                        <table className="cost-table">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th className="cost-table-cell-right">Cost</th>
                                    <th className="cost-table-cell-right">Calls</th>
                                    <th className="cost-table-cell-right">Input Tokens</th>
                                    <th className="cost-table-cell-right">Output Tokens</th>
                                </tr>
                            </thead>
                            <tbody>
                                {featureCosts.map((fc) => (
                                    <tr key={fc.task_type}>
                                        <td className="cost-table-cell-name">
                                            {formatTaskType(fc.task_type)}
                                        </td>
                                        <td className="cost-table-cell-right">
                                            {centsToUsd(fc.total_cost_cents)}
                                        </td>
                                        <td className="cost-table-cell-right">
                                            {fc.call_count.toLocaleString()}
                                        </td>
                                        <td className="cost-table-cell-right">
                                            {formatTokens(fc.total_input_tokens)}
                                        </td>
                                        <td className="cost-table-cell-right">
                                            {formatTokens(fc.total_output_tokens)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* ---- Daily Spend bar chart ---- */}
            <div className="card mb-lg">
                <h3 className="mb-md">Daily Spend</h3>

                {dailyCosts.length === 0 ? (
                    <p className="text-muted text-sm">No daily data for this period.</p>
                ) : (
                    <div className="cost-chart">
                        {dailyCosts.map((dc) => {
                            const heightPct =
                                maxDailyCost > 0
                                    ? (dc.cost_cents / maxDailyCost) * 100
                                    : 0
                            return (
                                <div
                                    key={dc.date}
                                    className="cost-chart-bar"
                                    title={`${dc.date}: ${centsToUsd(dc.cost_cents)}`}
                                    style={{ '--bar-height': `${Math.max(heightPct, 2)}%` } as React.CSSProperties}
                                />
                            )
                        })}
                    </div>
                )}

                {/* X-axis labels (first and last date) */}
                {dailyCosts.length > 1 && (
                    <div className="cost-chart-axis text-xs text-muted">
                        <span>{dailyCosts[0].date}</span>
                        <span>{dailyCosts[dailyCosts.length - 1].date}</span>
                    </div>
                )}
            </div>

            {/* ---- Budget Configuration form ---- */}
            <div className="card">
                <h3 className="mb-md">Budget Configuration</h3>

                {saveSuccess && (
                    <div className="card card-success mb-md">
                        <p className="success-message">{saveSuccess}</p>
                    </div>
                )}

                <form onSubmit={handleSaveBudget}>
                    <div className="form-field">
                        <label className="form-label" htmlFor="budget-cents">
                            Monthly Budget (cents)
                        </label>
                        <input
                            id="budget-cents"
                            className="form-input"
                            type="number"
                            min={0}
                            value={budgetCents}
                            onChange={(e) => setBudgetCents(Number(e.target.value))}
                            placeholder="e.g. 5000 for $50.00"
                        />
                        <span className="text-xs text-muted cost-form-hint">
                            {budgetCents > 0 ? `= ${centsToUsd(budgetCents)}` : 'Enter 0 to disable budget limit'}
                        </span>
                    </div>

                    <div className="form-field mt-md">
                        <label className="form-label" htmlFor="warning-pct">
                            Warning Threshold (%)
                        </label>
                        <input
                            id="warning-pct"
                            className="form-input"
                            type="number"
                            min={0}
                            max={100}
                            value={warningPct}
                            onChange={(e) => setWarningPct(Number(e.target.value))}
                            placeholder="e.g. 80"
                        />
                        <span className="text-xs text-muted cost-form-hint">
                            Alert triggers when spend exceeds this percentage of the budget
                        </span>
                    </div>

                    <div className="cost-budget-actions">
                        <button
                            type="submit"
                            className="btn-primary"
                            disabled={saving}
                        >
                            {saving ? 'Saving...' : 'Save'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}
