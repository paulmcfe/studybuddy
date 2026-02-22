/**
 * Benchmark Results for StudyBuddy v13
 *
 * Run and display performance benchmarks across models.
 */

import { useState, useEffect } from "react";
import { apiFetch } from "../lib/api";

interface BenchmarkResult {
  provider: string;
  model: string;
  latency_ms: number;
  response_length: number;
  tokens_per_second: number;
  success: boolean;
  error: string | null;
}

interface HistoricalResult {
  id: string;
  provider: string;
  model: string;
  latency_ms: number;
  tokens_per_second: number;
  response_length: number;
  success: boolean;
  error_message: string | null;
  created_at: string;
}

export default function BenchmarkResults() {
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [history, setHistory] = useState<HistoricalResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(
    "Explain spaced repetition in 3 sentences."
  );

  useEffect(() => {
    fetchHistory();
  }, []);

  async function fetchHistory() {
    setLoading(true);
    try {
      const res = await apiFetch("/api/benchmark/history?limit=20");
      const data = await res.json();
      setHistory(data.results || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  }

  async function runBenchmark() {
    setRunning(true);
    setError(null);
    setResults([]);

    try {
      const res = await apiFetch(
        `/api/benchmark/run?prompt=${encodeURIComponent(prompt)}`,
        { method: "POST" }
      );
      const data = await res.json();
      setResults(data.results || []);
      // Refresh history after running
      fetchHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed");
    } finally {
      setRunning(false);
    }
  }

  function formatLatency(ms: number): string {
    if (ms >= 1000) {
      return `${(ms / 1000).toFixed(2)}s`;
    }
    return `${ms.toFixed(0)}ms`;
  }

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-4">Performance Benchmarks</h2>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {/* Run benchmark */}
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Run Benchmark</h3>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Test Prompt</label>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full border rounded px-3 py-2"
            placeholder="Enter a test prompt..."
          />
        </div>

        <button
          onClick={runBenchmark}
          disabled={running || !prompt}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {running ? "Running Benchmark..." : "Run Benchmark"}
        </button>

        <p className="text-gray-500 text-sm mt-2">
          Rate limited to 5 runs per hour
        </p>
      </div>

      {/* Current results */}
      {results.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4">Current Results</h3>
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2">Model</th>
                <th className="text-left py-2">Provider</th>
                <th className="text-right py-2">Latency</th>
                <th className="text-right py-2">Tokens/sec</th>
                <th className="text-right py-2">Response Length</th>
                <th className="text-center py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {results
                .sort((a, b) => a.latency_ms - b.latency_ms)
                .map((result, idx) => (
                  <tr
                    key={idx}
                    className={`border-b ${
                      !result.success ? "bg-red-50" : idx === 0 ? "bg-green-50" : ""
                    }`}
                  >
                    <td className="py-2">{result.model}</td>
                    <td className="py-2 capitalize">{result.provider}</td>
                    <td className="py-2 text-right">
                      {result.success ? formatLatency(result.latency_ms) : "-"}
                    </td>
                    <td className="py-2 text-right">
                      {result.success
                        ? result.tokens_per_second.toFixed(1)
                        : "-"}
                    </td>
                    <td className="py-2 text-right">
                      {result.success ? result.response_length : "-"}
                    </td>
                    <td className="py-2 text-center">
                      {result.success ? (
                        <span className="text-green-600">OK</span>
                      ) : (
                        <span
                          className="text-red-600"
                          title={result.error || "Failed"}
                        >
                          Failed
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Historical results */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Recent History</h3>

        {loading ? (
          <div className="animate-pulse">Loading history...</div>
        ) : history.length === 0 ? (
          <p className="text-gray-500">No benchmark history yet.</p>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2">Date</th>
                <th className="text-left py-2">Model</th>
                <th className="text-left py-2">Provider</th>
                <th className="text-right py-2">Latency</th>
                <th className="text-right py-2">Tokens/sec</th>
                <th className="text-center py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {history.map((result) => (
                <tr
                  key={result.id}
                  className={`border-b ${!result.success ? "bg-red-50" : ""}`}
                >
                  <td className="py-2 text-sm">
                    {new Date(result.created_at).toLocaleString()}
                  </td>
                  <td className="py-2">{result.model}</td>
                  <td className="py-2 capitalize">{result.provider}</td>
                  <td className="py-2 text-right">
                    {result.success ? formatLatency(result.latency_ms) : "-"}
                  </td>
                  <td className="py-2 text-right">
                    {result.success
                      ? result.tokens_per_second?.toFixed(1) || "-"
                      : "-"}
                  </td>
                  <td className="py-2 text-center">
                    {result.success ? (
                      <span className="text-green-600">OK</span>
                    ) : (
                      <span
                        className="text-red-600"
                        title={result.error_message || "Failed"}
                      >
                        Failed
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
