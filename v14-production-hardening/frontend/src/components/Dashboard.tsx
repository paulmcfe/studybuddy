'use client'

import { useState, useEffect } from 'react'
import { apiFetch } from '../lib/api'

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
        recently_indexed?: boolean
    }
    flashcards: {
        total: number
        due: number
        mastered: number
        learning: number
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
    const [isUpdatingCurriculum, setIsUpdatingCurriculum] = useState(false)
    const [lastKnownDocCount, setLastKnownDocCount] = useState<number | null>(null)

    useEffect(() => {
        if (program?.id) {
            setLastKnownDocCount(null)
            setIsUpdatingCurriculum(false)

            const loadInitialData = async () => {
                const statsResponse = await apiFetch(`/api/programs/${program.id}/stats`)
                const statsData = await statsResponse.json()
                setStats(statsData)

                await loadCurriculum()
                setLoading(false)

                // Auto-generate curriculum if missing
                const topicCount = statsData?.topics?.total ?? 0
                if (topicCount === 0 && program.description) {
                    setIsGeneratingCurriculum(true)
                    try {
                        const response = await apiFetch(
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
                        }
                    } catch (error) {
                        console.error('Auto curriculum generation failed:', error)
                    } finally {
                        setIsGeneratingCurriculum(false)
                    }
                }
            }
            loadInitialData()
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

    const hasDocuments = (stats?.documents?.total ?? 0) > 0
    const hasCurriculum = (stats?.topics?.total ?? 0) > 0
    const isIndexingDocuments = (stats?.documents?.pending ?? 0) > 0
    const recentlyIndexed = stats?.documents?.recently_indexed ?? false
    const currentDocCount = stats?.documents?.indexed ?? 0

    // Detect when new documents are indexed and curriculum needs updating
    useEffect(() => {
        if (!hasDocuments || !hasCurriculum) return

        if (lastKnownDocCount !== null && currentDocCount !== lastKnownDocCount) {
            setIsUpdatingCurriculum(true)
        }
        if (isIndexingDocuments && !isUpdatingCurriculum) {
            setIsUpdatingCurriculum(true)
        }
        if (recentlyIndexed && !isUpdatingCurriculum) {
            setIsUpdatingCurriculum(true)
        }
        if (currentDocCount > 0) {
            setLastKnownDocCount(currentDocCount)
        }
    }, [currentDocCount, hasCurriculum, hasDocuments, lastKnownDocCount, isIndexingDocuments, isUpdatingCurriculum, recentlyIndexed])

    // Clear updating state after documents finish indexing and curriculum regenerates
    useEffect(() => {
        if (!isUpdatingCurriculum) return
        if (isIndexingDocuments) return
        if (recentlyIndexed) return

        const timer = setTimeout(async () => {
            await loadCurriculum()
            await loadStats()
            setIsUpdatingCurriculum(false)
        }, 5000)

        return () => clearTimeout(timer)
    }, [isUpdatingCurriculum, isIndexingDocuments, recentlyIndexed])

    // Poll while documents are being indexed or curriculum is updating
    useEffect(() => {
        if (!isIndexingDocuments && !isUpdatingCurriculum && !recentlyIndexed) return

        const pollInterval = setInterval(async () => {
            await loadCurriculum()
            await loadStats()
        }, 2000)

        return () => clearInterval(pollInterval)
    }, [program?.id, isIndexingDocuments, isUpdatingCurriculum, recentlyIndexed])

    const loadStats = async () => {
        try {
            const response = await apiFetch(`/api/programs/${program.id}/stats`)
            const data = await response.json()
            setStats(data)
        } catch (error) {
            console.error('Failed to load stats:', error)
        }
    }

    const loadCurriculum = async () => {
        try {
            const response = await apiFetch(`/api/programs/${program.id}`)
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
            const response = await apiFetch(`/api/programs/${program.id}`, {
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

            {(hasCurriculum && curriculum) || isUpdatingCurriculum ? (
                <div className="card mt-lg">
                    <div
                        className={`curriculum-toggle${isUpdatingCurriculum ? ' disabled' : ''}`}
                        onClick={() => {
                            if (!isUpdatingCurriculum) {
                                setShowCurriculum(!showCurriculum)
                            }
                        }}
                    >
                        <h3 className="card-title">
                            {isUpdatingCurriculum ? (
                                <>Updating curriculum...</>
                            ) : (
                                <>Curriculum ({stats?.topics?.total} topics)</>
                            )}
                        </h3>
                        <div className="flex align-center gap-sm">
                            {isUpdatingCurriculum ? (
                                <div className="spinner spinner-sm"></div>
                            ) : (
                                <span className="curriculum-arrow">
                                    {showCurriculum ? '▼' : '▶'}
                                </span>
                            )}
                        </div>
                    </div>

                    {showCurriculum && curriculum && (
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
            ) : isGeneratingCurriculum ? (
                <div className="card mt-lg">
                    <div className="card-header">
                        <div>
                            <h3 className="card-title">Generating curriculum...</h3>
                            <p className="card-subtitle">
                                {hasDocuments
                                    ? 'Creating a curriculum based on your uploaded documents.'
                                    : 'Creating a curriculum based on your program description.'}
                            </p>
                        </div>
                        <div className="spinner spinner-sm"></div>
                    </div>
                </div>
            ) : null}
        </div>
    )
}
