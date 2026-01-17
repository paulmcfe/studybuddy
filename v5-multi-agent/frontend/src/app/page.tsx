'use client'

import { useState, useEffect, useCallback } from 'react'
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

    // Get current chapter object
    const currentChapter = chapters.find((c) => c.id === selectedChapter) || null

    // Status message for home screen
    const getStatusMessage = () => {
        if (!agentStatus) return 'Connecting to agent...'
        if (!agentStatus.indexing_complete) {
            return `Indexing "${agentStatus.current_file}"... (${agentStatus.chunks_in_db} chunks)`
        }
        return `${agentStatus.documents_indexed} guides indexed (${agentStatus.chunks_in_db} chunks)`
    }

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
                    // Auto-select first chapter
                    if (data.chapters?.length > 0) {
                        setSelectedChapter(data.chapters[0].id)
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

    const handleGotIt = () => {
        // New random topic
        fetchFlashcard()
    }

    const handleStudyMore = () => {
        // Same topic, different question
        if (currentCard) {
            fetchFlashcard(currentCard.topic, currentCard.question)
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
                />
            )}

            {/* Main content */}
            <main className="flex-1 flex flex-col overflow-hidden">
                {currentScreen === 'home' ? (
                    <HomeScreen
                        chapters={chapters}
                        selectedChapter={selectedChapter}
                        scope={scope}
                        onChapterChange={setSelectedChapter}
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
                        onGotIt={handleGotIt}
                        onStudyMore={handleStudyMore}
                        onOpenChat={handleOpenChat}
                        onBack={handleBack}
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
