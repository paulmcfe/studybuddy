import { useState, useEffect, useRef } from 'react'

function DocumentManager({ program, onUpdate }) {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [dragover, setDragover] = useState(false)
  const fileInputRef = useRef(null)

  useEffect(() => {
    loadDocuments()
  }, [program.id])

  const loadDocuments = async () => {
    try {
      const response = await fetch(`/api/programs/${program.id}/documents`)
      const data = await response.json()
      setDocuments(data.documents)
    } catch (error) {
      console.error('Failed to load documents:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async (files) => {
    setUploading(true)

    for (const file of files) {
      const formData = new FormData()
      formData.append('file', file)

      try {
        const response = await fetch(`/api/programs/${program.id}/documents`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const error = await response.json()
          alert(`Failed to upload ${file.name}: ${error.detail}`)
        }
      } catch (error) {
        console.error('Upload failed:', error)
        alert(`Failed to upload ${file.name}`)
      }
    }

    setUploading(false)
    loadDocuments()
    onUpdate()
  }

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files)
    if (files.length > 0) {
      handleUpload(files)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragover(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleUpload(files)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragover(true)
  }

  const handleDragLeave = () => {
    setDragover(false)
  }

  const handleDelete = async (documentId) => {
    if (!confirm('Delete this document?')) return

    try {
      await fetch(`/api/programs/${program.id}/documents/${documentId}`, {
        method: 'DELETE',
      })
      loadDocuments()
      onUpdate()
    } catch (error) {
      console.error('Delete failed:', error)
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const getFileIcon = (fileType) => {
    if (fileType?.includes('pdf')) return '📄'
    if (fileType?.includes('markdown')) return '📝'
    return '📃'
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
      <div className="flex justify-between items-center" style={{ marginBottom: '1.5rem' }}>
        <h2>Documents: {program.name}</h2>
      </div>

      <div
        className={`upload-zone ${dragover ? 'dragover' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        style={{ marginBottom: '1.5rem' }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.md,.markdown,.txt"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        {uploading ? (
          <div className="loading">
            <div className="spinner"></div>
          </div>
        ) : (
          <>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📁</div>
            <p>Drop files here or click to upload</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>
              Supports PDF, Markdown, and plain text files (max 50MB)
            </p>
          </>
        )}
      </div>

      {documents.length === 0 ? (
        <div className="card empty-state">
          <div className="empty-state-icon">📚</div>
          <h3>No Documents Yet</h3>
          <p style={{ marginTop: '0.5rem' }}>
            Upload PDF, Markdown, or text files to build your knowledge base.
          </p>
        </div>
      ) : (
        <div className="document-list">
          {documents.map((doc) => (
            <div key={doc.id} className="document-item">
              <div className="document-info">
                <div className="document-icon">{getFileIcon(doc.file_type)}</div>
                <div>
                  <div className="document-name">{doc.filename}</div>
                  <div className="document-meta">
                    {formatFileSize(doc.file_size)}
                    {doc.chunks_count && ` · ${doc.chunks_count} chunks`}
                  </div>
                </div>
              </div>
              <div className="flex gap-md items-center">
                <span className={`document-status ${doc.status}`}>
                  {doc.status === 'indexed' && '✓ Indexed'}
                  {doc.status === 'pending' && '⏳ Pending'}
                  {doc.status === 'processing' && '🔄 Processing'}
                  {doc.status === 'failed' && '❌ Failed'}
                </span>
                <button
                  className="btn-secondary"
                  onClick={() => handleDelete(doc.id)}
                  style={{ padding: '0.25rem 0.5rem' }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {documents.some((d) => d.status === 'failed') && (
        <div
          className="card"
          style={{
            marginTop: '1rem',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderColor: 'var(--color-error)',
          }}
        >
          <p style={{ color: 'var(--color-error)' }}>
            Some documents failed to index. Try re-uploading them or check the file format.
          </p>
        </div>
      )}
    </div>
  )
}

export default DocumentManager
