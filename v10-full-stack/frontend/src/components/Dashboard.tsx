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

interface Topic {
    title: string
    subtopics: string[]
}

interface Chapter {
    number: number
    title: string
    topics: Topic[]
}

interface TopicList {
    chapters: Chapter[]
}

type View = 'overview' | 'study' | 'flashcards' | 'documents' | 'new-program'

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
    const [curriculum, setCurriculum] = useState<TopicList | null>(null)
    const [showCurriculum, setShowCurriculum] = useState(false)
    const [loading, setLoading] = useState(true)
    const [isDeleting, setIsDeleting] = useState(false)
    const [isGeneratingCurriculum, setIsGeneratingCurriculum] = useState(false)

    useEffect(() => {
        if (program?.id) {
            loadStats()
            loadCurriculum()
        }
    }, [program?.id])

    // Poll for updated stats while background flashcard generation might be in progress
    useEffect(() => {
        if ((stats?.flashcards?.total ?? 0) >= 6) return

        const pollInterval = setInterval(async () => {
            await loadStats()
            onProgramUpdate() // Refresh sidebar counts
        }, 3000) // Check every 3 seconds

        return () => clearInterval(pollInterval)
    }, [program?.id, stats?.flashcards?.total])

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

    const loadCurriculum = async () => {
        try {
            const response = await fetch(`/api/programs/${program.id}`)
            const data = await response.json()
            setCurriculum(data.topic_list || null)
        } catch (error) {
            console.error('Failed to load curriculum:', error)
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

    const handleGenerateCurriculum = async () => {
        if (!program.description) {
            alert('Program needs a description to generate a curriculum.')
            return
        }

        setIsGeneratingCurriculum(true)
        try {
            const response = await fetch(
                `/api/programs/${program.id}/generate-curriculum`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        topic: program.description,
                        depth: 'intermediate',
                        chapter_count: 8,
                    }),
                }
            )

            if (response.ok) {
                await loadCurriculum()
                await loadStats()
                onProgramUpdate()
            } else {
                const error = await response.json()
                alert(error.detail || 'Failed to generate curriculum')
            }
        } catch (error) {
            console.error('Failed to generate curriculum:', error)
            alert('Failed to generate curriculum')
        } finally {
            setIsGeneratingCurriculum(false)
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
            <div className="page-header">
                <div className="page-header-content">
                    <h2>{program.name}</h2>
                    {program.description && (
                        <p className="page-description">{program.description}</p>
                    )}
                </div>
                <div className="page-header-actions flex gap-sm">
                    <button onClick={() => onNavigate('flashcards')} className="btn-primary">
                        Start Studying
                    </button>
                </div>
            </div>

            <div className="card mb-lg">
                <h3 className="mb-md">Flashcard Progress</h3>
                <div className="stats-grid">
                    <div className="stat-card">
                        <div className="stat-value">{stats?.flashcards?.due || 0}</div>
                        <div className="stat-label">Ready to Review</div>
                        <div className="stat-sublabel">Cards to study now</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">{stats?.flashcards?.learning || 0}</div>
                        <div className="stat-label">Still Learning</div>
                        <div className="stat-sublabel">Building memory</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">{stats?.flashcards?.mastered || 0}</div>
                        <div className="stat-label">Mastered</div>
                        <div className="stat-sublabel">In long-term memory</div>
                    </div>
                </div>
            </div>

            <div className="card mb-lg">
                <h3 className="mb-md">Quick Actions</h3>
                <div className="flex gap-md">
                    <button onClick={() => onNavigate('documents')} className="btn-secondary">
                        Upload Documents
                    </button>
                    <button onClick={() => onNavigate('flashcards')} className="btn-secondary">
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

            {stats?.documents?.pending != null && stats.documents.pending > 0 && (
                <div className="card card-warning">
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
                    <p className="mt-sm mb-md">
                        Upload some documents to start building your knowledge base.
                    </p>
                    <button onClick={() => onNavigate('documents')} className="btn-primary">
                        Upload Documents
                    </button>
                </div>
            )}

            {(stats?.topics?.total ?? 0) > 0 && curriculum ? (
                <div className="card mt-lg">
                    <div
                        className="curriculum-toggle"
                        onClick={() => setShowCurriculum(!showCurriculum)}
                    >
                        <h3 className="card-title">
                            Curriculum ({stats.topics.total} topics)
                        </h3>
                        <span className="curriculum-arrow">
                            {showCurriculum ? '▼' : '▶'}
                        </span>
                    </div>

                    {showCurriculum && (
                        <div className="curriculum-content">
                            {curriculum.chapters?.map((chapter, i) => (
                                <div key={i} className="curriculum-chapter">
                                    <h4 className="curriculum-chapter-title">
                                        {chapter.title}
                                    </h4>
                                    <ul className="curriculum-topics">
                                        {chapter.topics?.map((topic, j) => (
                                            <li key={j} className="curriculum-topic">
                                                • {topic.title}
                                                {topic.subtopics?.length > 0 && (
                                                    <ul className="curriculum-subtopics">
                                                        {topic.subtopics.map((sub, k) => (
                                                            <li key={k} className="curriculum-subtopic">
                                                                ◦ {sub}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                )}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            ) : program.description && (
                <div className="card mt-lg">
                    <div className="card-header">
                        <div>
                            <h3 className="card-title">Curriculum</h3>
                            <p className="card-subtitle">
                                No curriculum yet. Generate one based on your program description.
                            </p>
                        </div>
                        <button
                            type="button"
                            className="btn-primary"
                            onClick={handleGenerateCurriculum}
                            disabled={isGeneratingCurriculum}
                        >
                            {isGeneratingCurriculum ? 'Generating...' : 'Generate Curriculum'}
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
