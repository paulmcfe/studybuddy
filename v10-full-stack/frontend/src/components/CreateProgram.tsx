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
        <div className="create-program-container">
            <h2 className="create-program-title">Create Learning Program</h2>

            <form onSubmit={handleCreate}>
                <div className="card form-section">
                    <div className="form-field">
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

                <div className={`card form-section ${canSelectCurriculum ? '' : 'card-disabled'}`}>
                    <h3 className="section-title">Curriculum</h3>

                    <div className="tabs mb-md">
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
                        <p className="text-secondary">
                            Upload your own documents and the AI tutor will use them directly.
                            Later, from the program overview, you can generate a curriculum based on your program description.
                        </p>
                    )}

                    {createMode === 'generate' && (
                        <p className="text-secondary">
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
