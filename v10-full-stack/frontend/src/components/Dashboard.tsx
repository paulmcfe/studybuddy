'use client'

import { useState, useEffect } from 'react'

interface Program {
    id: string
    name: string
    description?: string
}

interface Stats {
    documents: {
        indexed: number
        pending: number
        total: number
    }
    flashcards: {
        total: number
        due: number
        mastered: number
    }
    topics: {
        total: number
    }
}

type View = 'overview' | 'study' | 'documents' | 'new-program'

interface DashboardProps {
    program: Program
    onProgramUpdate: () => void
    onProgramDelete: (programId: string) => void
    onNavigate: (view: View) => void
}

export default function Dashboard({
    program,
    onProgramUpdate,
    onProgramDelete,
    onNavigate,
}: DashboardProps) {
    const [stats, setStats] = useState<Stats | null>(null)
    const [loading, setLoading] = useState(true)
    const [isDeleting, setIsDeleting] = useState(false)

    useEffect(() => {
        if (program?.id) {
            loadStats()
        }
    }, [program?.id])

    const loadStats = async () => {
        try {
            const response = await fetch(`/api/programs/${program.id}/stats`)
            const data = await response.json()
            setStats(data)
        } catch (error) {
            console.error('Failed to load stats:', error)
        } finally {
            setLoading(false)
        }
    }

    const handleDelete = async () => {
        if (!confirm(`Are you sure you want to delete "${program.name}"? This will remove all documents and flashcards.`)) {
            return
        }

        setIsDeleting(true)
        try {
            const response = await fetch(`/api/programs/${program.id}`, {
                method: 'DELETE',
            })

            if (response.ok) {
                onProgramDelete?.(program.id)
            } else {
                alert('Failed to delete program')
            }
        } catch (error) {
            console.error('Failed to delete program:', error)
            alert('Failed to delete program')
        } finally {
            setIsDeleting(false)
        }
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
            <div className="flex justify-between items-center" style={{ marginBottom: '2rem' }}>
                <div>
                    <h2>{program.name}</h2>
                    {program.description && (
                        <p style={{ color: 'var(--color-text-secondary)', marginTop: '0.5rem' }}>
                            {program.description}
                        </p>
                    )}
                </div>
                <div className="flex gap-sm">
                    <button onClick={() => onNavigate('study')} className="btn-primary">
                        Start Studying
                    </button>
                </div>
            </div>

            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-value">{stats?.documents?.indexed || 0}</div>
                    <div className="stat-label">Documents Indexed</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{stats?.flashcards?.total || 0}</div>
                    <div className="stat-label">Flashcards</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{stats?.flashcards?.due || 0}</div>
                    <div className="stat-label">Due for Review</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{stats?.flashcards?.mastered || 0}</div>
                    <div className="stat-label">Mastered</div>
                </div>
            </div>

            <div className="card" style={{ marginBottom: '1.5rem' }}>
                <h3 style={{ marginBottom: '1rem' }}>Quick Actions</h3>
                <div className="flex gap-md">
                    <button onClick={() => onNavigate('documents')} className="btn-secondary">
                        Upload Documents
                    </button>
                    <button onClick={() => onNavigate('study')} className="btn-secondary">
                        Review Flashcards
                    </button>
                    <button onClick={() => onNavigate('study')} className="btn-secondary">
                        Chat with Tutor
                    </button>
                    <button
                        className="btn-danger"
                        onClick={handleDelete}
                        disabled={isDeleting}
                    >
                        {isDeleting ? 'Deleting...' : 'Delete Program'}
                    </button>
                </div>
            </div>

            {stats?.documents?.pending && stats.documents.pending > 0 && (
                <div
                    className="card"
                    style={{
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        borderColor: 'var(--color-warning)',
                    }}
                >
                    <p>
                        <strong>{stats.documents.pending}</strong> document(s) are still being indexed.
                        They&apos;ll be available for studying soon.
                    </p>
                </div>
            )}

            {stats?.documents?.total === 0 && (
                <div className="card empty-state">
                    <div className="empty-state-icon">📚</div>
                    <h3>No Documents Yet</h3>
                    <p style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
                        Upload some documents to start building your knowledge base.
                    </p>
                    <button onClick={() => onNavigate('documents')} className="btn-primary">
                        Upload Documents
                    </button>
                </div>
            )}

            {stats?.topics?.total && stats.topics.total > 0 && (
                <div className="card" style={{ marginTop: '1.5rem' }}>
                    <h3 style={{ marginBottom: '1rem' }}>Topic List</h3>
                    <p style={{ color: 'var(--color-text-secondary)' }}>
                        {stats.topics.total} topics in your curriculum
                    </p>
                </div>
            )}
        </div>
    )
}
