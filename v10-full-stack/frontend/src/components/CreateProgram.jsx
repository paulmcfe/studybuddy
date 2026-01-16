import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

function CreateProgram({ onCreated }) {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [topicInput, setTopicInput] = useState('')
  const [generating, setGenerating] = useState(false)
  const [createMode, setCreateMode] = useState('manual') // manual, generate
  const [loading, setLoading] = useState(false)

  const handleCreate = async (e) => {
    e.preventDefault()
    if (!name.trim()) return

    setLoading(true)

    try {
      // Create the program
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

      // If generating curriculum, do that next
      if (createMode === 'generate' && topicInput.trim()) {
        setGenerating(true)

        const curriculumResponse = await fetch(
          `/api/programs/${program.id}/generate-curriculum`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              topic: topicInput.trim(),
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
      navigate('/')
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
            <label htmlFor="description">Description (optional)</label>
            <textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What do you want to learn?"
              rows={3}
            />
          </div>
        </div>

        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Curriculum</h3>

          <div className="tabs" style={{ marginBottom: '1rem' }}>
            <button
              type="button"
              className={`tab ${createMode === 'manual' ? 'active' : ''}`}
              onClick={() => setCreateMode('manual')}
            >
              Start Empty
            </button>
            <button
              type="button"
              className={`tab ${createMode === 'generate' ? 'active' : ''}`}
              onClick={() => setCreateMode('generate')}
            >
              Generate with AI
            </button>
          </div>

          {createMode === 'manual' && (
            <p style={{ color: 'var(--color-text-secondary)' }}>
              You can add a topic list later by uploading documents or generating
              one with AI.
            </p>
          )}

          {createMode === 'generate' && (
            <div>
              <label htmlFor="topic">What do you want to learn?</label>
              <input
                id="topic"
                type="text"
                value={topicInput}
                onChange={(e) => setTopicInput(e.target.value)}
                placeholder="e.g., Machine Learning, Spanish for beginners"
              />
              <p
                style={{
                  fontSize: '0.75rem',
                  color: 'var(--color-text-muted)',
                  marginTop: '0.5rem',
                }}
              >
                AI will generate a comprehensive curriculum for this topic
              </p>
            </div>
          )}
        </div>

        <div className="flex gap-md">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => navigate('/')}
            disabled={loading}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn-primary"
            disabled={!name.trim() || loading}
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

export default CreateProgram
