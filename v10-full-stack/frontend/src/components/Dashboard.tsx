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
    const [lastKnownDocCount, setLastKnownDocCount] = useState<number | null>(null)
    const [isUpdatingCurriculum, setIsUpdatingCurriculum] = useState(false)
    const [lastKnownTopicCount, setLastKnownTopicCount] = useState<number | null>(null)
    const [hadCurriculumBefore, setHadCurriculumBefore] = useState(false)

    useEffect(() => {
        if (program?.id) {
            // Reset tracking state when program changes
            setLastKnownDocCount(null)
            setIsUpdatingCurriculum(false)
            setLastKnownTopicCount(null)
            setHadCurriculumBefore(false)

            // Load data and check if curriculum already exists
            const loadInitialData = async () => {
                const statsResponse = await fetch(`/api/programs/${program.id}/stats`)
                const statsData = await statsResponse.json()
                setStats(statsData)

                // If curriculum already exists when we load, mark it as "had before"
                // This prevents showing "Updating curriculum..." for existing programs
                const existingTopicCount = statsData?.topics?.total ?? 0
                if (existingTopicCount > 0) {
                    setHadCurriculumBefore(true)
                }

                await loadCurriculum()
                setLoading(false)
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

    // Poll for curriculum while documents are being indexed (Document-Based workflow)
    const hasDocuments = (stats?.documents?.total ?? 0) > 0
    const hasCurriculum = (stats?.topics?.total ?? 0) > 0
    const isIndexingDocuments = (stats?.documents?.pending ?? 0) > 0
    const recentlyIndexed = stats?.documents?.recently_indexed ?? false
    const currentDocCount = stats?.documents?.indexed ?? 0
    const currentTopicCount = stats?.topics?.total ?? 0

    // Detect when curriculum is being generated for the first time from documents
    // This should only be true for initial generation, NOT for updates to existing curriculum
    // Once hadCurriculumBefore is true, we've moved past initial generation
    const isGeneratingCurriculumFromDocs = hasDocuments && !hasCurriculum && !hadCurriculumBefore

    // Track when we've seen a curriculum (to distinguish initial generation from updates)
    // Only set hadCurriculumBefore=true AFTER recentlyIndexed becomes false
    // This prevents showing "Updating curriculum..." during initial generation
    useEffect(() => {
        if (hasCurriculum && !hadCurriculumBefore && !recentlyIndexed && !isIndexingDocuments) {
            console.log('Dashboard: Curriculum stable (not recently indexed), setting hadCurriculumBefore=true')
            setHadCurriculumBefore(true)
        }
    }, [hasCurriculum, hadCurriculumBefore, recentlyIndexed, isIndexingDocuments])

    // Detect when curriculum might be updating (document count changed, or documents recently indexed)
    // Only trigger "Updating" state if we had a curriculum BEFORE (not during initial generation)
    // AND only for document-based programs (hasDocuments must be true)
    useEffect(() => {
        // Skip all update detection for programs without documents (AI-generated programs)
        // Curriculum updates only happen when documents are indexed
        if (!hasDocuments) {
            return
        }

        // Case 1: Document count changed while we have a curriculum that existed before
        if (lastKnownDocCount !== null && currentDocCount !== lastKnownDocCount && hasCurriculum && hadCurriculumBefore) {
            console.log('Dashboard: Document count changed, setting isUpdatingCurriculum=true')
            setIsUpdatingCurriculum(true)
        }
        // Case 2: There are pending documents and we have a curriculum that existed before
        if (isIndexingDocuments && hasCurriculum && hadCurriculumBefore && !isUpdatingCurriculum) {
            console.log('Dashboard: Documents indexing with existing curriculum, setting isUpdatingCurriculum=true')
            setIsUpdatingCurriculum(true)
        }
        // Case 3: Documents were recently indexed (within 30 seconds) - curriculum may still be regenerating
        // Only if we had a curriculum before (not during initial generation completing)
        if (recentlyIndexed && hasCurriculum && hadCurriculumBefore && !isUpdatingCurriculum) {
            console.log('Dashboard: Documents recently indexed, setting isUpdatingCurriculum=true')
            setIsUpdatingCurriculum(true)
        }
        if (currentDocCount > 0) {
            setLastKnownDocCount(currentDocCount)
        }
    }, [currentDocCount, hasCurriculum, hadCurriculumBefore, hasDocuments, lastKnownDocCount, isIndexingDocuments, isUpdatingCurriculum, recentlyIndexed])

    // Clear updating state when curriculum topic count changes (curriculum was regenerated)
    useEffect(() => {
        if (!isUpdatingCurriculum) {
            setLastKnownTopicCount(currentTopicCount)
            return
        }

        // If topic count changed while updating, the curriculum was regenerated
        if (lastKnownTopicCount !== null && currentTopicCount !== lastKnownTopicCount && !isIndexingDocuments) {
            console.log('Dashboard: Topic count changed, curriculum updated, clearing isUpdatingCurriculum')
            setIsUpdatingCurriculum(false)
            setLastKnownTopicCount(currentTopicCount)
            return
        }

        // Set initial topic count if not set
        if (lastKnownTopicCount === null && currentTopicCount > 0) {
            setLastKnownTopicCount(currentTopicCount)
        }
    }, [isUpdatingCurriculum, currentTopicCount, lastKnownTopicCount, isIndexingDocuments])

    // Timeout: Clear updating state after waiting for curriculum regeneration to complete
    useEffect(() => {
        if (!isUpdatingCurriculum) return
        if (isIndexingDocuments) return // Still indexing, wait longer
        if (recentlyIndexed) return // Recently indexed, curriculum might still be regenerating

        // Documents finished indexing and enough time has passed - clear the updating state
        const timer = setTimeout(() => {
            console.log('Dashboard: Timeout reached, clearing isUpdatingCurriculum')
            loadCurriculum().then(() => {
                setIsUpdatingCurriculum(false)
            })
        }, 5000) // 5 seconds after recentlyIndexed becomes false

        return () => clearTimeout(timer)
    }, [isUpdatingCurriculum, isIndexingDocuments, recentlyIndexed])

    useEffect(() => {
        // Poll while curriculum is being generated or updated from documents
        const shouldPoll = isGeneratingCurriculumFromDocs || isIndexingDocuments || isUpdatingCurriculum

        if (!shouldPoll) return

        console.log('Dashboard: Starting poll loop', { isGeneratingCurriculumFromDocs, isIndexingDocuments, isUpdatingCurriculum })

        const pollInterval = setInterval(async () => {
            await loadCurriculum()
            await loadStats()
        }, 2000) // Check every 2 seconds

        return () => {
            clearInterval(pollInterval)
        }
    }, [program?.id, isGeneratingCurriculumFromDocs, isIndexingDocuments, isUpdatingCurriculum])

    const loadStats = async () => {
        try {
            const response = await fetch(`/api/programs/${program.id}/stats`)
            const data = await response.json()
            setStats(data)
        } catch (error) {
            console.error('Failed to load stats:', error)
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

            {hasCurriculum && curriculum ? (
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
            ) : isGeneratingCurriculumFromDocs ? (
                <div className="card mt-lg">
                    <div className="card-header">
                        <div>
                            <h3 className="card-title">Generating curriculum...</h3>
                            <p className="card-subtitle">
                                Creating a curriculum based on your uploaded documents.
                            </p>
                        </div>
                        <div className="spinner spinner-sm"></div>
                    </div>
                </div>
            ) : program.description && !hasDocuments && (
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
