'use client'

import { useState } from 'react'

interface Program {
    id: string
    name: string
    description?: string
    document_count: number
    flashcard_count: number
}

interface CreateProgramProps {
    onCreated: (program: Program) => void
    onCancel: () => void
}

export default function CreateProgram({ onCreated, onCancel }: CreateProgramProps) {
    const [name, setName] = useState('')
    const [description, setDescription] = useState('')
    const [generating, setGenerating] = useState(false)
    const [createMode, setCreateMode] = useState<'manual' | 'generate'>('manual')
    const [loading, setLoading] = useState(false)

    const canSelectCurriculum = name.trim() && description.trim()

    const handleCreate = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!name.trim()) return

        setLoading(true)

        try {
            const response = await fetch('/api/programs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name.trim(),
                    description: description.trim() || null,
                }),
            })

            if (!response.ok) {
                throw new Error('Failed to create program')
            }

            const program = await response.json()

            if (createMode === 'generate') {
                setGenerating(true)

                const curriculumResponse = await fetch(
                    `/api/programs/${program.id}/generate-curriculum`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            topic: description.trim(),
                            depth: 'intermediate',
                            chapter_count: 8,
                        }),
                    }
                )

                if (!curriculumResponse.ok) {
                    console.error('Failed to generate curriculum')
                }
            }

            onCreated(program)
        } catch (error) {
            console.error('Error creating program:', error)
            alert('Failed to create program')
        } finally {
            setLoading(false)
            setGenerating(false)
        }
    }

    return (
        <div style={{ maxWidth: 600, margin: '0 auto' }}>
            <h2 style={{ marginBottom: '1.5rem' }}>Create Learning Program</h2>

            <form onSubmit={handleCreate}>
                <div className="card" style={{ marginBottom: '1.5rem' }}>
                    <div style={{ marginBottom: '1rem' }}>
                        <label htmlFor="name">Program Name *</label>
                        <input
                            id="name"
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g., Spanish Vocabulary, AWS Certification"
                            required
                        />
                    </div>

                    <div>
                        <label htmlFor="description">What do you want to learn? *</label>
                        <textarea
                            id="description"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="e.g., Conversational Spanish for travel, Machine Learning fundamentals"
                            rows={3}
                            required
                        />
                    </div>
                </div>

                <div
                    className="card"
                    style={{
                        marginBottom: '1.5rem',
                        opacity: canSelectCurriculum ? 1 : 0.5,
                        pointerEvents: canSelectCurriculum ? 'auto' : 'none',
                    }}
                >
                    <h3 style={{ marginBottom: '1rem' }}>Curriculum</h3>

                    <div className="tabs" style={{ marginBottom: '1rem' }}>
                        <button
                            type="button"
                            className={`tab ${createMode === 'manual' ? 'active' : ''}`}
                            onClick={() => setCreateMode('manual')}
                            disabled={!canSelectCurriculum}
                        >
                            Start Empty
                        </button>
                        <button
                            type="button"
                            className={`tab ${createMode === 'generate' ? 'active' : ''}`}
                            onClick={() => setCreateMode('generate')}
                            disabled={!canSelectCurriculum}
                        >
                            Generate with AI
                        </button>
                    </div>

                    {createMode === 'manual' && (
                        <p style={{ color: 'var(--color-text-secondary)' }}>
                            Upload your own documents and the AI tutor will use them directly.
                            You can generate a topic list later if needed.
                        </p>
                    )}

                    {createMode === 'generate' && (
                        <p style={{ color: 'var(--color-text-secondary)' }}>
                            AI will generate a structured curriculum based on your description above.
                        </p>
                    )}
                </div>

                <div className="flex gap-md">
                    <button
                        type="button"
                        className="btn-secondary"
                        onClick={onCancel}
                        disabled={loading}
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        className="btn-primary"
                        disabled={!canSelectCurriculum || loading}
                    >
                        {loading
                            ? generating
                                ? 'Generating Curriculum...'
                                : 'Creating...'
                            : 'Create Program'}
                    </button>
                </div>
            </form>
        </div>
    )
}
