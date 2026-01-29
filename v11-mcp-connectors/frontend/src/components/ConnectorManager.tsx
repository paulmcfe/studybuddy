'use client'

import { useState, useEffect } from 'react'

interface Program {
    id: string
    name: string
}

interface Connector {
    id: string
    connector_type: 'fetch' | 'github' | 'brave_search'
    name: string
    config: Record<string, any>
    status: string
    last_sync_at: string | null
    error_message: string | null
    created_at: string
    updated_at: string
}

interface GitHubFile {
    path: string
    type: string
    size: number
}

interface ConnectorManagerProps {
    program: Program
    onUpdate: () => void
}

type ActivePanel = 'fetch' | 'github' | 'brave_search' | null

export default function ConnectorManager({ program, onUpdate }: ConnectorManagerProps) {
    const [connectors, setConnectors] = useState<Connector[]>([])
    const [loading, setLoading] = useState(true)
    const [activePanel, setActivePanel] = useState<ActivePanel>(null)

    // Fetch/URL state
    const [fetchUrl, setFetchUrl] = useState('')
    const [fetchImporting, setFetchImporting] = useState(false)
    const [fetchMessage, setFetchMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

    // GitHub state
    const [ghOwner, setGhOwner] = useState('')
    const [ghRepo, setGhRepo] = useState('')
    const [ghToken, setGhToken] = useState('')
    const [ghBranch, setGhBranch] = useState('main')
    const [ghFiles, setGhFiles] = useState<GitHubFile[]>([])
    const [ghSelectedFiles, setGhSelectedFiles] = useState<Set<string>>(new Set())
    const [ghBrowsing, setGhBrowsing] = useState(false)
    const [ghImporting, setGhImporting] = useState(false)
    const [ghSaving, setGhSaving] = useState(false)
    const [ghMessage, setGhMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
    const [ghConnectorId, setGhConnectorId] = useState<string | null>(null)

    // Brave Search state
    const [braveApiKey, setBraveApiKey] = useState('')
    const [braveSaving, setBraveSaving] = useState(false)
    const [braveMessage, setBraveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

    // Environment variable availability
    const [envVars, setEnvVars] = useState<{ brave_api_key: boolean; github_token: boolean }>({ brave_api_key: false, github_token: false })

    useEffect(() => {
        loadConnectors()
    }, [program.id])

    const loadConnectors = async () => {
        try {
            const response = await fetch(`/api/programs/${program.id}/connectors`)
            const data = await response.json()
            setConnectors(data.connectors)
            if (data.env) setEnvVars(data.env)
        } catch (error) {
            console.error('Failed to load connectors:', error)
        } finally {
            setLoading(false)
        }
    }

    // ============ Fetch/URL Connector ============

    const getFetchConnector = () =>
        connectors.find((c) => c.connector_type === 'fetch')

    const ensureFetchConnector = async (): Promise<string> => {
        const existing = getFetchConnector()
        if (existing) return existing.id

        const response = await fetch(`/api/programs/${program.id}/connectors`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                connector_type: 'fetch',
                name: 'URL Import',
                config: {},
            }),
        })
        const data = await response.json()
        await loadConnectors()
        return data.id
    }

    const handleFetchImport = async () => {
        if (!fetchUrl.trim()) return

        setFetchImporting(true)
        setFetchMessage(null)

        try {
            const connectorId = await ensureFetchConnector()

            const response = await fetch(
                `/api/programs/${program.id}/connectors/${connectorId}/fetch`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: fetchUrl }),
                }
            )

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || 'Import failed')
            }

            const data = await response.json()
            setFetchMessage({ type: 'success', text: data.message })
            setFetchUrl('')
            onUpdate()
            loadConnectors()
            setTimeout(() => setFetchMessage(null), 5000)
        } catch (error: any) {
            setFetchMessage({ type: 'error', text: error.message })
        } finally {
            setFetchImporting(false)
        }
    }

    // ============ GitHub Connector ============

    const getGitHubConnector = () =>
        connectors.find((c) => c.connector_type === 'github')

    const handleGitHubSave = async () => {
        if (!ghOwner.trim() || !ghRepo.trim()) return
        if (!ghToken.trim() && !envVars.github_token) return

        setGhSaving(true)
        setGhMessage(null)

        try {
            const existing = getGitHubConnector()

            const ghConfig: Record<string, string> = {
                owner: ghOwner,
                repo: ghRepo,
                branch: ghBranch,
            }
            if (ghToken.trim()) {
                ghConfig.token = ghToken
            }

            if (existing) {
                // Update existing connector
                await fetch(
                    `/api/programs/${program.id}/connectors/${existing.id}`,
                    {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            name: `${ghOwner}/${ghRepo}`,
                            config: ghConfig,
                        }),
                    }
                )
                setGhConnectorId(existing.id)
            } else {
                // Create new connector
                const response = await fetch(
                    `/api/programs/${program.id}/connectors`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            connector_type: 'github',
                            name: `${ghOwner}/${ghRepo}`,
                            config: ghConfig,
                        }),
                    }
                )
                const data = await response.json()
                setGhConnectorId(data.id)
            }

            setGhMessage({ type: 'success', text: 'Repository configured. You can now browse and import files.' })
            await loadConnectors()
        } catch (error: any) {
            setGhMessage({ type: 'error', text: error.message })
        } finally {
            setGhSaving(false)
        }
    }

    const handleGitHubBrowse = async () => {
        const connector = getGitHubConnector()
        const connectorId = ghConnectorId || connector?.id
        if (!connectorId) return

        setGhBrowsing(true)
        setGhMessage(null)

        try {
            const response = await fetch(
                `/api/programs/${program.id}/connectors/${connectorId}/github/files?branch=${ghBranch}`
            )

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || 'Failed to browse files')
            }

            const data = await response.json()
            setGhFiles(data.files)
            setGhSelectedFiles(new Set())

            if (data.files.length === 0) {
                setGhMessage({ type: 'error', text: 'No importable files found (looking for .md, .txt, .rst, .adoc files)' })
            }
        } catch (error: any) {
            setGhMessage({ type: 'error', text: error.message })
        } finally {
            setGhBrowsing(false)
        }
    }

    const toggleFileSelection = (path: string) => {
        const updated = new Set(ghSelectedFiles)
        if (updated.has(path)) {
            updated.delete(path)
        } else {
            updated.add(path)
        }
        setGhSelectedFiles(updated)
    }

    const toggleAllFiles = () => {
        if (ghSelectedFiles.size === ghFiles.length) {
            setGhSelectedFiles(new Set())
        } else {
            setGhSelectedFiles(new Set(ghFiles.map((f) => f.path)))
        }
    }

    const handleGitHubImport = async () => {
        const connector = getGitHubConnector()
        const connectorId = ghConnectorId || connector?.id
        if (!connectorId || ghSelectedFiles.size === 0) return

        setGhImporting(true)
        setGhMessage(null)

        try {
            const response = await fetch(
                `/api/programs/${program.id}/connectors/${connectorId}/github/import`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        file_paths: Array.from(ghSelectedFiles),
                    }),
                }
            )

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || 'Import failed')
            }

            const data = await response.json()
            setGhMessage({ type: 'success', text: data.message })
            setGhSelectedFiles(new Set())
            onUpdate()
            loadConnectors()
            // Background task needs time to create documents — refresh again after delay
            setTimeout(() => {
                onUpdate()
                loadConnectors()
            }, 5000)
            setTimeout(() => setGhMessage(null), 5000)
        } catch (error: any) {
            setGhMessage({ type: 'error', text: error.message })
        } finally {
            setGhImporting(false)
        }
    }

    const handleGitHubSync = async () => {
        const connector = getGitHubConnector()
        if (!connector) return

        setGhImporting(true)
        setGhMessage(null)

        try {
            const response = await fetch(
                `/api/programs/${program.id}/connectors/${connector.id}/github/sync`,
                { method: 'POST' }
            )

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || 'Sync failed')
            }

            const data = await response.json()
            setGhMessage({ type: 'success', text: data.message })
            loadConnectors()
        } catch (error: any) {
            setGhMessage({ type: 'error', text: error.message })
        } finally {
            setGhImporting(false)
        }
    }

    // Initialize GitHub form from existing connector
    useEffect(() => {
        const gh = getGitHubConnector()
        if (gh && activePanel === 'github') {
            const config = gh.config || {}
            if (config.owner) setGhOwner(config.owner)
            if (config.repo) setGhRepo(config.repo)
            if (config.branch) setGhBranch(config.branch)
            setGhConnectorId(gh.id)
            // Don't populate the token since it's masked
        }
    }, [activePanel, connectors])

    // ============ Brave Search Connector ============

    const getBraveConnector = () =>
        connectors.find((c) => c.connector_type === 'brave_search')

    const handleBraveEnable = async () => {
        setBraveSaving(true)
        setBraveMessage(null)

        try {
            const existing = getBraveConnector()
            const apiKeyPayload = braveApiKey.trim() ? { api_key: braveApiKey } : {}

            if (existing) {
                await fetch(
                    `/api/programs/${program.id}/connectors/${existing.id}`,
                    {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            config: { ...apiKeyPayload, enabled: true },
                        }),
                    }
                )
            } else {
                const response = await fetch(
                    `/api/programs/${program.id}/connectors`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            connector_type: 'brave_search',
                            name: 'Brave Web Search',
                            config: { ...apiKeyPayload, enabled: true },
                        }),
                    }
                )

                if (!response.ok) {
                    const error = await response.json()
                    throw new Error(error.detail || 'Failed to save')
                }
            }

            setBraveMessage({ type: 'success', text: 'Web search enabled for this program.' })
            setBraveApiKey('')
            await loadConnectors()
            setTimeout(() => setBraveMessage(null), 5000)
        } catch (error: any) {
            setBraveMessage({ type: 'error', text: error.message })
        } finally {
            setBraveSaving(false)
        }
    }

    const handleBraveToggle = async () => {
        const connector = getBraveConnector()
        if (!connector) return

        const currentlyEnabled = connector.config?.enabled !== false

        try {
            await fetch(
                `/api/programs/${program.id}/connectors/${connector.id}`,
                {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        config: { enabled: !currentlyEnabled },
                    }),
                }
            )
            await loadConnectors()
        } catch (error) {
            console.error('Failed to toggle Brave Search:', error)
        }
    }

    // ============ Delete Connector ============

    const handleDeleteConnector = async (connectorId: string, withDocuments: boolean) => {
        const confirmMsg = withDocuments
            ? 'Delete this connector and all its imported documents?'
            : 'Remove this connector? Imported documents will be kept.'

        if (!confirm(confirmMsg)) return

        try {
            await fetch(
                `/api/programs/${program.id}/connectors/${connectorId}?delete_documents=${withDocuments}`,
                { method: 'DELETE' }
            )
            await loadConnectors()
            onUpdate()
        } catch (error) {
            console.error('Failed to delete connector:', error)
        }
    }

    // ============ Helpers ============

    const formatDate = (dateStr: string | null) => {
        if (!dateStr) return 'Never'
        return new Date(dateStr).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        })
    }

    const getConnectorStatus = (connector: Connector) => {
        // Show meaningful status per connector type
        if (connector.connector_type === 'fetch') {
            return { label: 'Ready', className: 'document-status indexed' }
        }
        if (connector.connector_type === 'brave_search') {
            const enabled = connector.config?.enabled !== false
            return enabled
                ? { label: 'Enabled', className: 'document-status indexed' }
                : { label: 'Disabled', className: 'document-status pending' }
        }
        // GitHub: show sync status
        const statusMap: Record<string, { label: string; className: string }> = {
            configured: { label: 'Not synced', className: 'document-status pending' },
            syncing: { label: 'Syncing...', className: 'document-status pending' },
            synced: { label: 'Synced', className: 'document-status indexed' },
            failed: { label: 'Failed', className: 'document-status failed' },
        }
        return statusMap[connector.status] || { label: connector.status, className: 'document-status' }
    }

    const getConnectorLabel = (type: string) => {
        switch (type) {
            case 'fetch': return 'URL Import'
            case 'github': return 'GitHub'
            case 'brave_search': return 'Web Search'
            default: return type
        }
    }

    const getConnectorDisplayName = (connector: Connector) => {
        // Only show name if it adds info beyond the type badge
        const label = getConnectorLabel(connector.connector_type)
        if (connector.name === label || connector.name === 'URL Import' || connector.name === 'Brave Web Search') {
            return null
        }
        return connector.name
    }

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
            </div>
        )
    }

    return (
        <div>
            <div className="section-header">
                <h2>Connectors: {program.name}</h2>
            </div>

            {/* Add Connector Cards */}
            <div className="connector-cards">
                <div
                    className={`connector-card ${activePanel === 'fetch' ? 'active' : ''}`}
                    onClick={() => setActivePanel(activePanel === 'fetch' ? null : 'fetch')}
                >
                    <div className="connector-card-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="2" y1="12" x2="22" y2="12"/>
                            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                        </svg>
                    </div>
                    <h4 className="connector-card-title">Import from URL</h4>
                    <p className="connector-card-desc">
                        Import web pages, documentation, or articles as learning materials
                    </p>
                </div>

                <div
                    className={`connector-card ${activePanel === 'github' ? 'active' : ''}`}
                    onClick={() => setActivePanel(activePanel === 'github' ? null : 'github')}
                >
                    <div className="connector-card-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
                        </svg>
                    </div>
                    <h4 className="connector-card-title">Import from GitHub</h4>
                    <p className="connector-card-desc">
                        Import markdown files and docs from a GitHub repository
                    </p>
                </div>

                <div
                    className={`connector-card ${activePanel === 'brave_search' ? 'active' : ''}`}
                    onClick={() => setActivePanel(activePanel === 'brave_search' ? null : 'brave_search')}
                >
                    <div className="connector-card-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="11" cy="11" r="8"/>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                        </svg>
                    </div>
                    <h4 className="connector-card-title">Web Search</h4>
                    <p className="connector-card-desc">
                        Let the tutor search the web when documents don't have the answer
                    </p>
                </div>
            </div>

            {/* Fetch/URL Panel */}
            {activePanel === 'fetch' && (
                <div className="connector-panel">
                    <h3 className="connector-panel-title">Import from URL</h3>
                    <p className="text-secondary text-sm mb-md">
                        Enter a URL to fetch its content as a learning document. The page will be
                        converted to markdown and indexed for your study program.
                    </p>

                    <div className="connector-form-row">
                        <input
                            type="url"
                            value={fetchUrl}
                            onChange={(e) => setFetchUrl(e.target.value)}
                            placeholder="https://docs.example.com/tutorial"
                            className="connector-url-input"
                            onKeyDown={(e) => e.key === 'Enter' && handleFetchImport()}
                        />
                        <button
                            className="btn-primary"
                            onClick={handleFetchImport}
                            disabled={fetchImporting || !fetchUrl.trim()}
                        >
                            {fetchImporting ? 'Importing...' : 'Import'}
                        </button>
                    </div>

                    {fetchMessage && (
                        <div className={`connector-message ${fetchMessage.type}`}>
                            {fetchMessage.text}
                        </div>
                    )}
                </div>
            )}

            {/* GitHub Panel */}
            {activePanel === 'github' && (
                <div className="connector-panel">
                    <h3 className="connector-panel-title">Import from GitHub</h3>
                    <p className="text-secondary text-sm mb-md">
                        Connect a GitHub repository to import markdown documentation.
                        {!envVars.github_token && ' Requires a Personal Access Token with repo read access.'}
                    </p>

                    <div className="connector-form">
                        <div className="connector-form-group">
                            <label>Repository</label>
                            <div className="connector-form-row">
                                <input
                                    type="text"
                                    value={ghOwner}
                                    onChange={(e) => setGhOwner(e.target.value)}
                                    placeholder="owner"
                                />
                                <span className="connector-separator">/</span>
                                <input
                                    type="text"
                                    value={ghRepo}
                                    onChange={(e) => setGhRepo(e.target.value)}
                                    placeholder="repo"
                                />
                            </div>
                        </div>

                        {envVars.github_token ? (
                            <div className="connector-form-group">
                                <label>Personal Access Token</label>
                                <p className="text-sm text-muted">Using token from server environment.</p>
                            </div>
                        ) : (
                            <div className="connector-form-group">
                                <label>Personal Access Token</label>
                                <input
                                    type="password"
                                    value={ghToken}
                                    onChange={(e) => setGhToken(e.target.value)}
                                    placeholder="ghp_..."
                                />
                            </div>
                        )}

                        <div className="connector-form-group">
                            <label>Branch</label>
                            <input
                                type="text"
                                value={ghBranch}
                                onChange={(e) => setGhBranch(e.target.value)}
                                placeholder="main"
                            />
                        </div>

                        <div className="connector-form-actions">
                            <button
                                className="btn-primary"
                                onClick={handleGitHubSave}
                                disabled={ghSaving || !ghOwner.trim() || !ghRepo.trim() || (!ghToken.trim() && !envVars.github_token)}
                            >
                                {ghSaving ? 'Saving...' : 'Save Configuration'}
                            </button>
                            <button
                                className="btn-secondary"
                                onClick={handleGitHubBrowse}
                                disabled={ghBrowsing || (!ghConnectorId && !getGitHubConnector())}
                            >
                                {ghBrowsing ? 'Browsing...' : 'Browse Files'}
                            </button>
                        </div>
                    </div>

                    {ghMessage && (
                        <div className={`connector-message ${ghMessage.type}`}>
                            {ghMessage.text}
                        </div>
                    )}

                    {/* File Browser */}
                    {ghFiles.length > 0 && (
                        <div className="connector-file-browser">
                            <div className="connector-file-header">
                                <label className="connector-checkbox-label">
                                    <input
                                        type="checkbox"
                                        checked={ghSelectedFiles.size === ghFiles.length}
                                        onChange={toggleAllFiles}
                                    />
                                    <span>Select all ({ghFiles.length} files)</span>
                                </label>
                                <button
                                    className="btn-primary btn-sm"
                                    onClick={handleGitHubImport}
                                    disabled={ghImporting || ghSelectedFiles.size === 0}
                                >
                                    {ghImporting
                                        ? 'Importing...'
                                        : `Import ${ghSelectedFiles.size} file${ghSelectedFiles.size !== 1 ? 's' : ''}`}
                                </button>
                            </div>
                            <div className="connector-file-list">
                                {ghFiles.map((file) => (
                                    <label key={file.path} className="connector-file-item">
                                        <input
                                            type="checkbox"
                                            checked={ghSelectedFiles.has(file.path)}
                                            onChange={() => toggleFileSelection(file.path)}
                                        />
                                        <span className="connector-file-path">{file.path}</span>
                                        {file.size > 0 && (
                                            <span className="connector-file-size">
                                                {(file.size / 1024).toFixed(1)} KB
                                            </span>
                                        )}
                                    </label>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Brave Search Panel */}
            {activePanel === 'brave_search' && (
                <div className="connector-panel">
                    <h3 className="connector-panel-title">Web Search (Brave)</h3>
                    <p className="text-secondary text-sm mb-md">
                        When enabled, the tutor can search the web for additional context
                        when your uploaded documents don't cover a topic. Responses will
                        include citations with source URLs.
                    </p>

                    {getBraveConnector() ? (
                        <div className="connector-brave-status">
                            <div className="connector-brave-toggle">
                                <span className="font-medium">
                                    Web Search is {getBraveConnector()?.config?.enabled !== false ? 'enabled' : 'disabled'}
                                </span>
                                <button
                                    className={getBraveConnector()?.config?.enabled !== false ? 'btn-secondary' : 'btn-primary'}
                                    onClick={handleBraveToggle}
                                >
                                    {getBraveConnector()?.config?.enabled !== false ? 'Disable' : 'Enable'}
                                </button>
                            </div>
                        </div>
                    ) : envVars.brave_api_key ? (
                        <div>
                            <p className="text-sm mb-md">
                                Brave Search API key detected from server environment.
                            </p>
                            <button
                                className="btn-primary"
                                onClick={handleBraveEnable}
                                disabled={braveSaving}
                            >
                                {braveSaving ? 'Enabling...' : 'Enable Web Search'}
                            </button>
                        </div>
                    ) : (
                        <div className="connector-form">
                            <div className="connector-form-group">
                                <label>Brave Search API Key</label>
                                <input
                                    type="password"
                                    value={braveApiKey}
                                    onChange={(e) => setBraveApiKey(e.target.value)}
                                    placeholder="BSA..."
                                />
                            </div>
                            <button
                                className="btn-primary"
                                onClick={handleBraveEnable}
                                disabled={braveSaving || !braveApiKey.trim()}
                            >
                                {braveSaving ? 'Saving...' : 'Enable Web Search'}
                            </button>
                        </div>
                    )}

                    {braveMessage && (
                        <div className={`connector-message ${braveMessage.type}`}>
                            {braveMessage.text}
                        </div>
                    )}
                </div>
            )}

            {/* Active Connectors List */}
            {connectors.length > 0 && (
                <div className="connector-active-section">
                    <h3 className="section-title">Active Connectors</h3>
                    <div className="connector-list">
                        {connectors.map((connector) => {
                            const status = getConnectorStatus(connector)
                            const displayName = getConnectorDisplayName(connector)
                            return (
                            <div key={connector.id} className="connector-item">
                                <div className="connector-item-info">
                                    <div className="connector-item-header">
                                        <span className="connector-type-badge">
                                            {getConnectorLabel(connector.connector_type)}
                                        </span>
                                        {displayName && (
                                            <span className="connector-item-name">
                                                {displayName}
                                            </span>
                                        )}
                                        <span className={status.className}>{status.label}</span>
                                    </div>
                                    {(connector.last_sync_at || connector.error_message) && (
                                    <div className="connector-item-meta">
                                        {connector.last_sync_at && (
                                            <span className="text-xs text-muted">
                                                Last sync: {formatDate(connector.last_sync_at)}
                                            </span>
                                        )}
                                        {connector.error_message && (
                                            <span className="text-xs text-error">
                                                {connector.error_message}
                                            </span>
                                        )}
                                    </div>
                                    )}
                                </div>
                                <div className="connector-item-actions">
                                    {connector.connector_type === 'github' && connector.status === 'synced' && (
                                        <button
                                            className="btn-secondary btn-sm"
                                            onClick={() => handleGitHubSync()}
                                        >
                                            Sync
                                        </button>
                                    )}
                                    <button
                                        className="btn-secondary btn-sm"
                                        onClick={() => handleDeleteConnector(
                                            connector.id,
                                            connector.connector_type !== 'brave_search'
                                        )}
                                    >
                                        Remove
                                    </button>
                                </div>
                            </div>
                            )
                        })}
                    </div>
                </div>
            )}

            {/* Empty State */}
            {connectors.length === 0 && !activePanel && (
                <div className="card empty-state">
                    <div className="empty-state-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
                            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
                        </svg>
                    </div>
                    <h3>No Connectors Yet</h3>
                    <p className="mt-sm">
                        Connect external sources to import learning materials automatically.
                        Choose a connector above to get started.
                    </p>
                </div>
            )}
        </div>
    )
}
