'use client'

import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import Flashcard from './Flashcard'

interface Program {
    id: string
    name: string
    document_count?: number
    flashcard_count?: number
}

interface Message {
    role: 'user' | 'assistant'
    content: string
}

interface Card {
    id: string
    topic: string
    question: string
    answer: string
    interval: number
    repetitions: number
}

interface StudyInterfaceProps {
    program: Program
    onUpdate?: () => void
    initialView?: 'chat' | 'flashcards'
}

// Convert LaTeX delimiters from \( \) and \[ \] to $ and $$
function convertLatexDelimiters(text: string): string {
    return text
        // Convert display math \[ ... \] to $$ ... $$
        .replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$')
        // Convert inline math \( ... \) to $ ... $
        .replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$')
}

export default function StudyInterface({ program, onUpdate, initialView = 'chat' }: StudyInterfaceProps) {
    const [view, setView] = useState<'chat' | 'flashcards'>(initialView)
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState('')
    const [isConnected, setIsConnected] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const [dueCards, setDueCards] = useState<Card[]>([])
    const [currentCardIndex, setCurrentCardIndex] = useState(0)
    const [isGenerating, setIsGenerating] = useState(false)
    const [totalCards, setTotalCards] = useState(0)
    const [reviewFeedback, setReviewFeedback] = useState<string | null>(null)
    const wsRef = useRef<WebSocket | null>(null)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        connectWebSocket()
        loadDueCards()

        return () => {
            if (wsRef.current) {
                wsRef.current.close()
            }
        }
    }, [program.id])

    // Poll for new cards while background generation might be in progress
    useEffect(() => {
        // Only poll if we have few cards (background generation likely in progress)
        // Don't poll while actively generating a card to avoid race conditions
        if (totalCards >= 6 || isGenerating) return

        const pollInterval = setInterval(() => {
            loadDueCards()
        }, 3000) // Check every 3 seconds

        return () => clearInterval(pollInterval)
    }, [program.id, totalCards, isGenerating])

    // Generate batch of cards when down to 1, so user sees progress (6 → 5 → 4 → ... → 1)
    useEffect(() => {
        if (dueCards.length === 1 && !isGenerating && view === 'flashcards') {
            generateFlashcards(5)
        }
    }, [dueCards.length, isGenerating, view])

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
            onUpdate?.()
        } catch (error) {
            console.error('Failed to load due cards:', error)
        }
    }

    const generateFlashcards = async (count: number = 1) => {
        if (isGenerating) return // Prevent double generation
        setIsGenerating(true)
        try {
            // Generate multiple cards in parallel
            const promises = Array.from({ length: count }, () =>
                fetch(`/api/programs/${program.id}/flashcards/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({}),
                })
            )

            const responses = await Promise.all(promises)
            const newCards: Card[] = []

            for (const response of responses) {
                if (response.ok) {
                    const card = await response.json()
                    newCards.push(card)
                }
            }

            if (newCards.length > 0) {
                // Add new cards to end of queue
                setDueCards(prev => {
                    const existingIds = new Set(prev.map(c => c.id))
                    const uniqueNewCards = newCards.filter(c => !existingIds.has(c.id))
                    return [...prev, ...uniqueNewCards]
                })
                setTotalCards(prev => prev + newCards.length)
                onUpdate?.()
            } else if (count === 1) {
                // Only show error for single card generation (user-initiated)
                alert('Failed to generate flashcard')
            }
        } catch (error) {
            console.error('Failed to generate flashcards:', error)
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

        wsRef.current?.send(
            JSON.stringify({
                type: 'chat',
                message: message,
            })
        )
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const formatInterval = (days: number): string => {
        if (days <= 1) return 'tomorrow'
        if (days < 7) return `in ${days} days`
        if (days < 14) return 'in about a week'
        if (days < 30) return `in ${Math.round(days / 7)} weeks`
        if (days < 60) return 'in about a month'
        return `in ${Math.round(days / 30)} months`
    }

    const handleReview = async (quality: number) => {
        const card = dueCards[currentCardIndex]
        if (!card) return

        try {
            const response = await fetch(`/api/programs/${program.id}/flashcards/${card.id}/review`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ quality }),
            })

            const result = await response.json()
            let feedback: string
            if (quality === 5) {
                feedback = `Got it! You'll see this again ${formatInterval(result.interval)}`
            } else if (quality === 3) {
                feedback = `Keep at it! You'll see this again ${formatInterval(result.interval)}`
            } else {
                feedback = `No problem! You'll see this again ${formatInterval(result.interval)}`
            }
            setReviewFeedback(feedback)

            // Clear feedback after 2 seconds
            setTimeout(() => setReviewFeedback(null), 2000)

            // Remove the reviewed card from local state immediately
            const remainingCards = dueCards.filter((_, i) => i !== currentCardIndex)
            setDueCards(remainingCards)

            if (remainingCards.length === 0) {
                // No more due cards - generate a new batch
                setCurrentCardIndex(0)
                await generateFlashcards(5)
            } else if (currentCardIndex >= remainingCards.length) {
                // Was at end, wrap to beginning
                setCurrentCardIndex(0)
            }
            // Otherwise index stays same, now pointing to next card
        } catch (error) {
            console.error('Failed to record review:', error)
        }
    }

    return (
        <div>
            <div className="section-header">
                <h2>Study: {program.name}</h2>
                <div className="tabs tabs-inline">
                    <button
                        className={`tab ${view === 'flashcards' ? 'active' : ''}`}
                        onClick={() => setView('flashcards')}
                    >
                        Flashcards ({dueCards.length})
                    </button>
                    <button
                        className={`tab ${view === 'chat' ? 'active' : ''}`}
                        onClick={() => setView('chat')}
                    >
                        Chat
                    </button>
                </div>
            </div>

            {view === 'chat' && (
                <div className="chat-container">
                    <div className="chat-messages">
                        {messages.length === 0 && (
                            <div className="empty-state">
                                <p>Ask questions about {program.name}!</p>
                                <p className="text-sm mt-sm">
                                    {program.document_count && program.document_count > 0
                                        ? 'The tutor will use your uploaded documents to answer.'
                                        : 'The tutor will use its general knowledge to help you learn.'}
                                </p>
                            </div>
                        )}
                        {messages.map((msg, i) => (
                            <div key={i} className={`chat-message ${msg.role}`}>
                                {msg.role === 'assistant' ? (
                                    <ReactMarkdown
                                        remarkPlugins={[remarkMath]}
                                        rehypePlugins={[rehypeKatex]}
                                    >
                                        {convertLatexDelimiters(msg.content)}
                                    </ReactMarkdown>
                                ) : (
                                    msg.content
                                )}
                            </div>
                        ))}
                        {isLoading && (
                            <div className="chat-message assistant">
                                <div className="spinner spinner-sm"></div>
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
                    <div className="card mb-lg">
                        <div className="card-header">
                            <div>
                                <h3 className="card-title">Flashcards</h3>
                                <p className="card-subtitle">
                                    {totalCards} cards total · {dueCards.length} due for review
                                </p>
                            </div>
                            <div className={`review-feedback${reviewFeedback ? '' : ' hidden'}`}>
                                {reviewFeedback || '\u00A0'}
                            </div>
                        </div>
                    </div>

                    {dueCards.length === 0 ? (
                        <div className="card empty-state">
                            <div className="empty-state-icon">✓</div>
                            <h3>No cards due for review</h3>
                            <p className="mt-sm">
                                {totalCards === 0
                                    ? 'Click "Generate New Card" to create your first flashcard!'
                                    : 'All caught up! Generate more cards or check back later.'}
                            </p>
                        </div>
                    ) : (
                        <>
                            <h3 className="mb-md">Review Due Cards</h3>
                            <div className="card-counter">
                                {dueCards.length} {dueCards.length === 1 ? 'card' : 'cards'} left to review
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
