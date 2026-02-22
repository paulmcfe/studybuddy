/**
 * Model Settings for StudyBuddy v13
 *
 * Configure per-task model selection with fallback chains.
 */

import { useState, useEffect } from "react";
import { apiFetch } from "../lib/api";

interface ProviderStatus {
  configured?: boolean;
  installed?: boolean;
  running?: boolean;
  base_url?: string;
}

interface ModelInfo {
  provider: string;
  display_name: string;
  model_id: string;
  context_length: number;
  input_cost_per_1m: number;
  output_cost_per_1m: number;
  best_for: string[];
  is_available: boolean;
  is_embedding: boolean;
}

interface TaskConfig {
  primary_provider: string;
  primary_model: string;
  fallback_provider: string | null;
  fallback_model: string | null;
  temperature: number;
  max_tokens: number | null;
  is_default?: boolean;
}

const TASK_LABELS: Record<string, string> = {
  flashcard_generation: "Flashcard Generation",
  tutoring: "Tutoring / Chat",
  curriculum: "Curriculum Generation",
  embedding: "Embeddings",
};

export default function ModelSettings() {
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [providers, setProviders] = useState<Record<string, ProviderStatus>>({});
  const [configs, setConfigs] = useState<Record<string, TaskConfig>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  async function fetchData() {
    setLoading(true);
    setError(null);

    try {
      const [modelsRes, configsRes] = await Promise.all([
        apiFetch("/api/models"),
        apiFetch("/api/models/config"),
      ]);

      const modelsData = await modelsRes.json();
      const configsData = await configsRes.json();
      setModels(modelsData.models || {});
      setProviders(modelsData.providers || {});
      setConfigs(configsData.configs || {});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load settings");
    } finally {
      setLoading(false);
    }
  }

  async function saveConfig(taskType: string) {
    setSaving(taskType);
    setError(null);
    setSuccess(null);

    try {
      const config = configs[taskType];
      await apiFetch("/api/models/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_type: taskType,
          primary_provider: config.primary_provider,
          primary_model: config.primary_model,
          fallback_provider: config.fallback_provider,
          fallback_model: config.fallback_model,
          temperature: config.temperature,
          max_tokens: config.max_tokens,
        }),
      });
      setSuccess(`${TASK_LABELS[taskType]} configuration saved!`);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(null);
    }
  }

  function updateConfig(
    taskType: string,
    field: keyof TaskConfig,
    value: string | number | null
  ) {
    setConfigs((prev) => ({
      ...prev,
      [taskType]: {
        ...prev[taskType],
        [field]: value,
        is_default: false,
      },
    }));
  }

  function getModelsForProvider(provider: string): [string, ModelInfo][] {
    return Object.entries(models).filter(
      ([_, m]) => m.provider === provider && !m.is_embedding
    );
  }

  if (loading) {
    return (
      <div className="p-6">
        <h2 className="text-xl font-bold mb-4">Model Settings</h2>
        <div className="animate-pulse">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-4">Model Settings</h2>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {success && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
          {success}
        </div>
      )}

      {/* Provider status */}
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Provider Status</h3>
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(providers).map(([name, status]) => (
            <div key={name} className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  name === "openai"
                    ? status.configured
                      ? "bg-green-500"
                      : "bg-red-500"
                    : name === "together"
                    ? status.configured
                      ? "bg-green-500"
                      : "bg-yellow-500"
                    : status.running
                    ? "bg-green-500"
                    : "bg-red-500"
                }`}
              />
              <span className="capitalize">{name}</span>
              <span className="text-gray-500 text-sm">
                {name === "openai"
                  ? status.configured
                    ? "(Ready)"
                    : "(Not configured)"
                  : name === "together"
                  ? status.configured
                    ? "(Ready)"
                    : "(No API key)"
                  : status.running
                  ? "(Running)"
                  : "(Not running)"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Task configurations */}
      {Object.entries(configs).map(([taskType, config]) => (
        <div key={taskType} className="bg-white rounded-lg shadow p-6 mb-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">
              {TASK_LABELS[taskType] || taskType}
            </h3>
            {config.is_default && (
              <span className="text-xs bg-gray-200 px-2 py-1 rounded">
                Default
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4 mb-4">
            {/* Primary model */}
            <div>
              <label className="block text-sm font-medium mb-1">
                Primary Provider
              </label>
              <select
                value={config.primary_provider}
                onChange={(e) =>
                  updateConfig(taskType, "primary_provider", e.target.value)
                }
                title="Primary Provider"
                className="w-full border rounded px-3 py-2"
              >
                <option value="openai">OpenAI</option>
                <option value="together">Together AI</option>
                <option value="ollama">Ollama (Local)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Primary Model
              </label>
              <select
                value={config.primary_model}
                onChange={(e) =>
                  updateConfig(taskType, "primary_model", e.target.value)
                }
                title="Primary Model"
                className="w-full border rounded px-3 py-2"
              >
                {getModelsForProvider(config.primary_provider).map(
                  ([key, model]) => (
                    <option
                      key={key}
                      value={model.model_id}
                      disabled={!model.is_available}
                    >
                      {model.display_name}{" "}
                      {!model.is_available && "(unavailable)"}
                    </option>
                  )
                )}
              </select>
            </div>

            {/* Fallback model */}
            <div>
              <label className="block text-sm font-medium mb-1">
                Fallback Provider
              </label>
              <select
                value={config.fallback_provider || ""}
                onChange={(e) =>
                  updateConfig(
                    taskType,
                    "fallback_provider",
                    e.target.value || null
                  )
                }
                title="Fallback Provider"
                className="w-full border rounded px-3 py-2"
              >
                <option value="">None</option>
                <option value="openai">OpenAI</option>
                <option value="together">Together AI</option>
                <option value="ollama">Ollama (Local)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Fallback Model
              </label>
              <select
                value={config.fallback_model || ""}
                title="Fallback Model"
                onChange={(e) =>
                  updateConfig(
                    taskType,
                    "fallback_model",
                    e.target.value || null
                  )
                }
                className="w-full border rounded px-3 py-2"
                disabled={!config.fallback_provider}
              >
                <option value="">None</option>
                {config.fallback_provider &&
                  getModelsForProvider(config.fallback_provider).map(
                    ([key, model]) => (
                      <option
                        key={key}
                        value={model.model_id}
                        disabled={!model.is_available}
                      >
                        {model.display_name}
                      </option>
                    )
                  )}
              </select>
            </div>
          </div>

          {/* Temperature slider */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">
              Temperature: {config.temperature}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              title="Temperature"
              value={config.temperature}
              onChange={(e) =>
                updateConfig(taskType, "temperature", parseFloat(e.target.value))
              }
              className="w-full"
            />
          </div>

          <button
            onClick={() => saveConfig(taskType)}
            disabled={saving === taskType}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
          >
            {saving === taskType ? "Saving..." : "Save Configuration"}
          </button>
        </div>
      ))}
    </div>
  );
}
