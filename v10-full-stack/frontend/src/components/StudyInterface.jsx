import { useState, useEffect, useRef } from 'react'
import Flashcard from './Flashcard'

function StudyInterface({ program }) {
  const [view, setView] = useState('chat') // chat, flashcards
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [dueCards, setDueCards] = useState([])
  const [currentCardIndex, setCurrentCardIndex] = useState(0)
  const [isGenerating, setIsGenerating] = useState(false)
  const [totalCards, setTotalCards] = useState(0)
  const [lastGeneratedCard, setLastGeneratedCard] = useState(null)
  const wsRef = useRef(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    connectWebSocket()
    loadDueCards()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [program.id])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const connectWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/study/${program.id}`

    wsRef.current = new WebSocket(wsUrl)

    wsRef.current.onopen = () => {
      setIsConnected(true)
    }

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'response') {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.content },
        ])
        setIsLoading(false)
      } else if (data.type === 'status') {
        // Could show status updates
      } else if (data.type === 'error') {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Error: ${data.message}` },
        ])
        setIsLoading(false)
      }
    }

    wsRef.current.onclose = () => {
      setIsConnected(false)
      // Reconnect after delay
      setTimeout(connectWebSocket, 3000)
    }

    wsRef.current.onerror = () => {
      setIsConnected(false)
    }
  }

  const loadDueCards = async () => {
    try {
      const response = await fetch(`/api/programs/${program.id}/due-cards?limit=20`)
      const data = await response.json()
      setDueCards(data.cards)
      setTotalCards(data.total_cards || 0)
    } catch (error) {
      console.error('Failed to load due cards:', error)
    }
  }

  const generateFlashcard = async () => {
    setIsGenerating(true)
    setLastGeneratedCard(null)
    try {
      const response = await fetch(`/api/programs/${program.id}/flashcards/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),  // Empty body - auto-select topic
      })

      if (response.ok) {
        const card = await response.json()
        setLastGeneratedCard(card)
        // Reload due cards to include the new one
        await loadDueCards()
      } else {
        const error = await response.json()
        console.error('Failed to generate flashcard:', error)
        alert(error.detail || 'Failed to generate flashcard')
      }
    } catch (error) {
      console.error('Failed to generate flashcard:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  const sendMessage = () => {
    if (!input.trim() || !isConnected || isLoading) return

    const message = input.trim()
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: message }])
    setIsLoading(true)

    wsRef.current.send(
      JSON.stringify({
        type: 'chat',
        message: message,
      })
    )
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleReview = async (quality) => {
    const card = dueCards[currentCardIndex]
    if (!card) return

    try {
      await fetch(`/api/programs/${program.id}/flashcards/${card.id}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ quality }),
      })

      // Move to next card or generate a new one
      if (currentCardIndex < dueCards.length - 1) {
        setCurrentCardIndex(currentCardIndex + 1)
      } else {
        // No more due cards - automatically generate a new one
        setCurrentCardIndex(0)
        await generateFlashcard()
      }
    } catch (error) {
      console.error('Failed to record review:', error)
    }
  }

  return (
    <div>
      <div className="flex justify-between items-center" style={{ marginBottom: '1.5rem' }}>
        <h2>Study: {program.name}</h2>
        <div className="tabs" style={{ border: 'none', marginBottom: 0 }}>
          <button
            className={`tab ${view === 'chat' ? 'active' : ''}`}
            onClick={() => setView('chat')}
          >
            Chat
          </button>
          <button
            className={`tab ${view === 'flashcards' ? 'active' : ''}`}
            onClick={() => setView('flashcards')}
          >
            Flashcards ({dueCards.length})
          </button>
        </div>
      </div>

      {view === 'chat' && (
        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <p>Ask questions about your learning materials!</p>
                <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                  The tutor will use your uploaded documents to answer.
                </p>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`chat-message ${msg.role}`}>
                {msg.content}
              </div>
            ))}
            {isLoading && (
              <div className="chat-message assistant">
                <div className="spinner" style={{ width: 20, height: 20 }}></div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-input-container">
            <input
              type="text"
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isConnected
                  ? 'Ask a question...'
                  : 'Connecting...'
              }
              disabled={!isConnected || isLoading}
            />
            <button
              className="btn-primary"
              onClick={sendMessage}
              disabled={!isConnected || isLoading || !input.trim()}
            >
              Send
            </button>
          </div>
        </div>
      )}

      {view === 'flashcards' && (
        <div>
          {/* Generate new flashcard section */}
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div>
                <h3 style={{ margin: 0 }}>Flashcards</h3>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0 0' }}>
                  {totalCards} cards total · {dueCards.length} due for review
                </p>
              </div>
              <button
                className="btn-primary"
                onClick={generateFlashcard}
                disabled={isGenerating}
              >
                {isGenerating ? 'Generating...' : 'Generate New Card'}
              </button>
            </div>

            {/* Show last generated card */}
            {lastGeneratedCard && (
              <div style={{
                padding: '1rem',
                background: 'var(--success-bg, #e8f5e9)',
                borderRadius: '8px',
                border: '1px solid var(--success-border, #c8e6c9)',
              }}>
                <p style={{ margin: '0 0 0.5rem 0', fontWeight: 500, color: 'var(--success-text, #2e7d32)' }}>
                  Card generated for: {lastGeneratedCard.topic}
                </p>
                <p style={{ margin: 0, fontSize: '0.875rem' }}>
                  <strong>Q:</strong> {lastGeneratedCard.question}
                </p>
              </div>
            )}
          </div>

          {/* Review section */}
          {dueCards.length === 0 ? (
            <div className="card empty-state">
              <div className="empty-state-icon">✓</div>
              <h3>No cards due for review</h3>
              <p style={{ marginTop: '0.5rem' }}>
                {totalCards === 0
                  ? 'Click "Generate New Card" to create your first flashcard!'
                  : 'All caught up! Generate more cards or check back later.'}
              </p>
            </div>
          ) : (
            <>
              <h3 style={{ marginBottom: '1rem' }}>Review Due Cards</h3>
              <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
                Card {currentCardIndex + 1} of {dueCards.length}
              </div>
              <Flashcard
                card={dueCards[currentCardIndex]}
                onReview={handleReview}
              />
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default StudyInterface
