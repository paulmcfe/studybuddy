/**
 * Cost Dashboard for StudyBuddy v13
 *
 * Displays token usage and cost breakdown by provider/model.
 */

import { useState, useEffect } from "react";
import { apiFetch } from "../lib/api";

interface CostBreakdown {
  provider: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  cost_cents: number;
  cost_usd: number;
  call_count: number;
  fallback_count: number;
}

interface CostSummary {
  period_days: number;
  total_cost_cents: number;
  total_cost_usd: number;
  breakdown: CostBreakdown[];
}

interface CostComparison {
  model: string;
  display_name: string;
  provider: string;
  input_cost: number;
  output_cost: number;
  total_cost: number;
  input_cost_per_1m: number;
  output_cost_per_1m: number;
}

interface CostComparisonResponse {
  input_tokens: number;
  output_tokens: number;
  comparison: CostComparison[];
}

export default function CostDashboard() {
  const [summary, setSummary] = useState<CostSummary | null>(null);
  const [comparison, setComparison] = useState<CostComparisonResponse | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState(30);

  useEffect(() => {
    fetchData();
  }, [days]);

  async function fetchData() {
    setLoading(true);
    setError(null);

    try {
      const [summaryRes, comparisonRes] = await Promise.all([
        apiFetch(`/api/costs?days=${days}`),
        apiFetch("/api/costs/comparison"),
      ]);

      setSummary(await summaryRes.json());
      setComparison(await comparisonRes.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load cost data");
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="p-6">
        <h2 className="text-xl font-bold mb-4">Cost Dashboard</h2>
        <div className="animate-pulse">Loading cost data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <h2 className="text-xl font-bold mb-4">Cost Dashboard</h2>
        <div className="text-red-500">{error}</div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-4">Cost Dashboard</h2>

      {/* Period selector */}
      <div className="mb-6">
        <label className="mr-2">Time period:</label>
        <select
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          title="Time period"
          className="border rounded px-2 py-1"
        >
          <option value={7}>Last 7 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
          <option value={365}>Last year</option>
        </select>
      </div>

      {/* Summary card */}
      {summary && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h3 className="text-lg font-semibold mb-2">Your Usage</h3>
          <div className="text-3xl font-bold text-blue-600">
            ${summary.total_cost_usd.toFixed(2)}
          </div>
          <div className="text-gray-500">Total cost over {summary.period_days} days</div>
        </div>
      )}

      {/* Usage breakdown */}
      {summary && summary.breakdown.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4">Usage by Model</h3>
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2">Provider</th>
                <th className="text-left py-2">Model</th>
                <th className="text-right py-2">Calls</th>
                <th className="text-right py-2">Input Tokens</th>
                <th className="text-right py-2">Output Tokens</th>
                <th className="text-right py-2">Cost</th>
              </tr>
            </thead>
            <tbody>
              {summary.breakdown.map((row, idx) => (
                <tr key={idx} className="border-b">
                  <td className="py-2 capitalize">{row.provider}</td>
                  <td className="py-2">{row.model}</td>
                  <td className="py-2 text-right">{row.call_count}</td>
                  <td className="py-2 text-right">
                    {row.input_tokens.toLocaleString()}
                  </td>
                  <td className="py-2 text-right">
                    {row.output_tokens.toLocaleString()}
                  </td>
                  <td className="py-2 text-right">${row.cost_usd.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Provider comparison */}
      {comparison && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-2">Provider Comparison</h3>
          <p className="text-gray-500 text-sm mb-4">
            Estimated monthly cost for {(comparison.input_tokens / 1_000_000).toFixed(0)}M
            input + {(comparison.output_tokens / 1_000_000).toFixed(0)}M output tokens
          </p>
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2">Model</th>
                <th className="text-left py-2">Provider</th>
                <th className="text-right py-2">Input $/1M</th>
                <th className="text-right py-2">Output $/1M</th>
                <th className="text-right py-2">Est. Monthly</th>
              </tr>
            </thead>
            <tbody>
              {comparison.comparison.map((row, idx) => (
                <tr
                  key={idx}
                  className={`border-b ${idx === 0 ? "bg-green-50" : ""}`}
                >
                  <td className="py-2">{row.display_name}</td>
                  <td className="py-2 capitalize">{row.provider}</td>
                  <td className="py-2 text-right">
                    ${row.input_cost_per_1m.toFixed(2)}
                  </td>
                  <td className="py-2 text-right">
                    ${row.output_cost_per_1m.toFixed(2)}
                  </td>
                  <td className="py-2 text-right font-semibold">
                    ${row.total_cost.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
