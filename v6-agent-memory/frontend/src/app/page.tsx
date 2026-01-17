'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import HomeScreen from '../components/HomeScreen'
import StudyScreen from '../components/StudyScreen'
import ChatPanel from '../components/ChatPanel'
import Sidebar from '../components/Sidebar'

// Types
interface Chapter {
    id: number
    title: string
    sections: { name: string; subtopics: string[] }[]
}

interface Flashcard {
    question: string
    answer: string
    topic: string
    source: 'rag' | 'llm'
    flashcard_id?: string
}

interface ChatMessage {
    id: string
    content: string
    role: 'user' | 'assistant'
    isLoading?: boolean
}

interface AgentStatus {
    indexing_complete: boolean
    documents_indexed: number
    chunks_in_db: number
    current_file: string
}

interface PrefetchStatus {
    state: 'idle' | 'in_progress' | 'completed'
    completed_topics: number
    total_topics: number
    cards_generated: number
}

export default function Home() {
    // Navigation state
    const [currentScreen, setCurrentScreen] = useState<'home' | 'study'>('home')

    // Chapter selection state
    const [chapters, setChapters] = useState<Chapter[]>([])
    const [selectedChapter, setSelectedChapter] = useState<number | null>(null)
    const [scope, setScope] = useState<'single' | 'cumulative'>('single')

    // Flashcard state
    const [currentCard, setCurrentCard] = useState<Flashcard | null>(null)
    const [isFlipped, setIsFlipped] = useState(false)
    const [isLoading, setIsLoading] = useState(false)

    // Chat state
    const [chatOpen, setChatOpen] = useState(false)
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
    const [isChatLoading, setIsChatLoading] = useState(false)

    // Status state
    const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null)
    const [statusMessage, setStatusMessage] = useState<string>('')
    const statusTimeoutRef = useRef<NodeJS.Timeout | null>(null)

    // Prefetch state
    const [prefetchStatus, setPrefetchStatus] = useState<Record<number, PrefetchStatus>>({})

    // Get current chapter object
    const currentChapter = chapters.find((c) => c.id === selectedChapter) || null

    // Show status message with auto-clear
    const showStatus = useCallback((message: string, duration = 3000) => {
        if (statusTimeoutRef.current) {
            clearTimeout(statusTimeoutRef.current)
        }
        setStatusMessage(message)
        if (duration > 0) {
            statusTimeoutRef.current = setTimeout(() => {
                setStatusMessage('')
            }, duration)
        }
    }, [])

    // Status message for home screen (indexing or prefetch progress)
    const getStatusMessage = useCallback(() => {
        if (!agentStatus) return 'Connecting to agent...'
        if (!agentStatus.indexing_complete) {
            return `Indexing "${agentStatus.current_file}"... (${agentStatus.chunks_in_db} chunks)`
        }
        // Check for active prefetch
        if (selectedChapter && prefetchStatus[selectedChapter]?.state === 'in_progress') {
            const ps = prefetchStatus[selectedChapter]
            return `Caching cards: ${ps.completed_topics}/${ps.total_topics} topics...`
        }
        // Show persistent status message if set
        if (statusMessage) {
            return statusMessage
        }
        return `${agentStatus.documents_indexed} guides indexed (${agentStatus.chunks_in_db} chunks)`
    }, [agentStatus, selectedChapter, prefetchStatus, statusMessage])

    // Poll status until indexing complete
    useEffect(() => {
        const checkStatus = async () => {
            try {
                const res = await fetch('/api/status')
                if (res.ok) {
                    const data = await res.json()
                    setAgentStatus(data)
                    return data.indexing_complete
                }
            } catch (err) {
                console.error('The status check failed:', err)
            }
            return false
        }

        const poll = async () => {
            const complete = await checkStatus()
            if (!complete) {
                setTimeout(poll, 2000)
            }
        }

        poll()
    }, [])

    // Fetch chapters when agent is ready
    useEffect(() => {
        if (agentStatus?.indexing_complete && chapters.length === 0) {
            fetch('/api/chapters')
                .then((res) => res.json())
                .then((data) => {
                    setChapters(data.chapters || [])
                    // Auto-select first chapter and trigger prefetch
                    if (data.chapters?.length > 0) {
                        const firstChapterId = data.chapters[0].id
                        setSelectedChapter(firstChapterId)
                        triggerPrefetch(firstChapterId)
                    }
                })
                .catch((err) => console.error('Failed to load chapters:', err))
        }
    }, [agentStatus?.indexing_complete, chapters.length])

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Don't trigger if typing in input
            if (
                e.target instanceof HTMLInputElement ||
                e.target instanceof HTMLTextAreaElement
            ) {
                return
            }

            if (e.code === 'Space' && currentScreen === 'study' && !chatOpen) {
                // Prevent page scroll
                e.preventDefault()
                // Only flip if the flashcard itself isn't handling this event
                // (flashcard has role="button" and handles Space via onKeyDown)
                const target = e.target as HTMLElement
                if (!target.closest('.flashcard')) {
                    setIsFlipped((prev) => !prev)
                }
            }

            if (e.code === 'Escape' && chatOpen) {
                setChatOpen(false)
            }
        }

        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [currentScreen, chatOpen])

    // Background prefetch
    const triggerPrefetch = async (chapterId: number) => {
        try {
            const res = await fetch('/api/prefetch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chapter_id: chapterId }),
            })
            if (res.ok) {
                const data = await res.json()
                if (data.state === 'started' || data.state === 'already_in_progress') {
                    pollPrefetchStatus(chapterId)
                }
            }
        } catch (err) {
            // Prefetch is non-critical, silently fail
            console.log('Prefetch request failed (non-critical):', err)
        }
    }

    const pollPrefetchStatus = async (chapterId: number) => {
        try {
            const res = await fetch(`/api/prefetch-status/${chapterId}`)
            if (res.ok) {
                const status = await res.json()
                setPrefetchStatus((prev) => ({ ...prev, [chapterId]: status }))
                if (status.state === 'in_progress') {
                    setTimeout(() => pollPrefetchStatus(chapterId), 2000)
                } else if (status.state === 'completed') {
                    showStatus(`${status.cards_generated} cards cached`)
                }
            }
        } catch (err) {
            console.log('Prefetch status poll failed:', err)
        }
    }

    // Record review to the API
    const recordReview = async (flashcardId: string, button: 'no' | 'took_a_sec' | 'yes') => {
        try {
            const res = await fetch('/api/review', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ flashcard_id: flashcardId, button }),
            })
            if (res.ok) {
                const data = await res.json()
                const days = data.interval_days
                const msg = days === 1
                    ? "You'll see this card again in 1 day"
                    : `You'll see this card again in ${days} days`
                showStatus(msg)
            }
        } catch (err) {
            console.error('Failed to record review:', err)
        }
    }

    // Fetch flashcard
    const fetchFlashcard = useCallback(
        async (currentTopic?: string, previousQuestion?: string) => {
            if (!selectedChapter) return

            setIsLoading(true)
            setIsFlipped(false)

            try {
                const res = await fetch('/api/flashcard', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        chapter_id: selectedChapter,
                        scope,
                        current_topic: currentTopic,
                        previous_question: previousQuestion,
                    }),
                })

                if (res.ok) {
                    const data = await res.json()
                    setCurrentCard(data)
                } else {
                    console.error('Flashcard fetch failed:', res.status)
                }
            } catch (err) {
                console.error('Flashcard fetch error:', err)
            } finally {
                setIsLoading(false)
            }
        },
        [selectedChapter, scope]
    )

    // Handlers
    const handleStartStudy = () => {
        setCurrentScreen('study')
        fetchFlashcard()
    }

    const handleBack = () => {
        setCurrentScreen('home')
        setCurrentCard(null)
        setChatMessages([])
        setChatOpen(false)
    }

    const handleChapterChange = (chapterId: number | null) => {
        setSelectedChapter(chapterId)
        // Trigger prefetch when chapter is selected
        if (chapterId) {
            triggerPrefetch(chapterId)
        }
    }

    const handleReview = async (button: 'no' | 'took_a_sec' | 'yes') => {
        // Record the review if we have a flashcard_id
        if (currentCard?.flashcard_id) {
            await recordReview(currentCard.flashcard_id, button)
        }

        // Fetch next card based on review result
        if (button === 'no') {
            // Same topic, different question (like old Study More)
            if (currentCard) {
                fetchFlashcard(currentCard.topic, currentCard.question)
            }
        } else {
            // New random topic (like old Got It)
            fetchFlashcard()
        }
    }

    const handleFlip = () => {
        setIsFlipped((prev) => !prev)
    }

    const handleOpenChat = () => {
        setChatOpen(true)
    }

    const handleCloseChat = () => {
        setChatOpen(false)
    }

    const handleSendMessage = async (message: string) => {
        // Add user message
        const userMsgId = `user-${Date.now()}`
        const assistantMsgId = `assistant-${Date.now()}`

        setChatMessages((prev) => [
            ...prev,
            { id: userMsgId, content: message, role: 'user' },
            { id: assistantMsgId, content: '', role: 'assistant', isLoading: true },
        ])
        setIsChatLoading(true)

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    chapter_id: selectedChapter,
                    scope,
                    card_context: currentCard
                        ? {
                            question: currentCard.question,
                            answer: currentCard.answer,
                            topic: currentCard.topic,
                        }
                        : undefined,
                }),
            })

            if (res.ok) {
                const data = await res.json()
                setChatMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === assistantMsgId
                            ? {
                                ...msg,
                                content: data.reply,
                                isLoading: false,
                            }
                            : msg
                    )
                )
            } else {
                setChatMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === assistantMsgId
                            ? {
                                ...msg,
                                content: 'Sorry, I encountered an error. Please try again.',
                                isLoading: false,
                            }
                            : msg
                    )
                )
            }
        } catch (err) {
            console.error('Chat error:', err)
            setChatMessages((prev) =>
                prev.map((msg) =>
                    msg.id === assistantMsgId
                        ? {
                            ...msg,
                            content: 'Sorry, I encountered an error. Please try again.',
                            isLoading: false,
                        }
                        : msg
                )
            )
        } finally {
            setIsChatLoading(false)
        }
    }

    return (
        <div className="flex h-screen bg-white dark:bg-[hsl(220,15%,13%)]">
            {/* Sidebar - desktop only, visible during study */}
            {currentScreen === 'study' && (
                <Sidebar
                    chapter={currentChapter}
                    scope={scope}
                    agentStatus={agentStatus}
                    statusMessage={statusMessage}
                />
            )}

            {/* Main content */}
            <main className="flex-1 flex flex-col overflow-hidden">
                {currentScreen === 'home' ? (
                    <HomeScreen
                        chapters={chapters}
                        selectedChapter={selectedChapter}
                        scope={scope}
                        onChapterChange={handleChapterChange}
                        onScopeChange={setScope}
                        onStart={handleStartStudy}
                        disabled={!agentStatus?.indexing_complete}
                        statusMessage={getStatusMessage()}
                    />
                ) : (
                    <StudyScreen
                        card={currentCard}
                        isFlipped={isFlipped}
                        isLoading={isLoading}
                        onFlip={handleFlip}
                        onReview={handleReview}
                        onOpenChat={handleOpenChat}
                        onBack={handleBack}
                        statusMessage={statusMessage}
                    />
                )}
            </main>

            {/* Chat panel */}
            <ChatPanel
                open={chatOpen}
                messages={chatMessages}
                onClose={handleCloseChat}
                onSendMessage={handleSendMessage}
                isLoading={isChatLoading}
            />
        </div>
    )
}
