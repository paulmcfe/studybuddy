'use client'

import { useState, useEffect, useCallback } from 'react'
import HomeScreen from '../components/HomeScreen'
import StudyScreen from '../components/StudyScreen'
import ChatPanel from '../components/ChatPanel'
import Sidebar from '../components/Sidebar'
import CurriculumModal from '../components/CurriculumModal'

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

interface CurriculumModule {
    id: string
    title: string
    description?: string
    estimated_hours?: number
    topics?: string[]
    status?: 'pending' | 'in_progress' | 'completed'
}

interface Curriculum {
    id: string
    goal: string
    modules: CurriculumModule[]
    currentModule: CurriculumModule | null
}

type StudyMode = 'chapter' | 'curriculum'

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
    const [chatPrefill, setChatPrefill] = useState('')

    // Status state
    const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null)

    // Curriculum state (v7)
    const [studyMode, setStudyMode] = useState<StudyMode>('chapter')
    const [activeCurriculum, setActiveCurriculum] = useState<Curriculum | null>(null)
    const [curriculumModalOpen, setCurriculumModalOpen] = useState(false)
    const [focusAreas, setFocusAreas] = useState<string[]>([])
    const [isCreatingCurriculum, setIsCreatingCurriculum] = useState(false)

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

    // Load saved curriculum from localStorage on mount
    useEffect(() => {
        const saved = localStorage.getItem('studybuddy-curriculum')
        if (saved) {
            try {
                const curriculum = JSON.parse(saved) as Curriculum
                setActiveCurriculum(curriculum)
            } catch (err) {
                console.error('Failed to parse saved curriculum:', err)
                localStorage.removeItem('studybuddy-curriculum')
            }
        }
    }, [])

    // Fetch focus areas (struggle topics) when agent is ready
    const fetchFocusAreas = useCallback(async () => {
        try {
            const res = await fetch('/api/stats')
            if (res.ok) {
                const data = await res.json()
                setFocusAreas(data.struggle_areas || [])
            }
        } catch (err) {
            console.error('Failed to fetch focus areas:', err)
        }
    }, [])

    useEffect(() => {
        if (agentStatus?.indexing_complete) {
            fetchFocusAreas()
        }
    }, [agentStatus?.indexing_complete, fetchFocusAreas])

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

    // Fetch curriculum flashcard
    const fetchCurriculumFlashcard = useCallback(
        async (currentTopic?: string, previousQuestion?: string) => {
            if (!activeCurriculum) return

            setIsLoading(true)
            setIsFlipped(false)

            try {
                const res = await fetch('/api/curriculum/flashcard', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        curriculum_id: activeCurriculum.id,
                        current_topic: currentTopic,
                        previous_question: previousQuestion,
                    }),
                })

                if (res.ok) {
                    const data = await res.json()
                    setCurrentCard(data)
                } else {
                    console.error('Curriculum flashcard fetch failed:', res.status)
                }
            } catch (err) {
                console.error('Curriculum flashcard fetch error:', err)
            } finally {
                setIsLoading(false)
            }
        },
        [activeCurriculum]
    )

    // Create curriculum
    const handleCreateCurriculum = async (goal: string, weeklyHours: number): Promise<Curriculum | null> => {
        setIsCreatingCurriculum(true)

        try {
            const res = await fetch('/api/curriculum', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ goal, weekly_hours: weeklyHours }),
            })

            if (res.ok) {
                const data = await res.json()
                // API returns { curriculum_id, curriculum: { goal, modules, ... } }
                const curriculumData = data.curriculum || {}
                const curriculum: Curriculum = {
                    id: data.curriculum_id,
                    goal: curriculumData.goal || goal,
                    modules: curriculumData.modules || [],
                    currentModule: curriculumData.modules?.[0] || null,
                }
                setActiveCurriculum(curriculum)
                localStorage.setItem('studybuddy-curriculum', JSON.stringify(curriculum))
                return curriculum
            } else {
                console.error('Curriculum creation failed:', res.status)
                return null
            }
        } catch (err) {
            console.error('Curriculum creation error:', err)
            return null
        } finally {
            setIsCreatingCurriculum(false)
        }
    }

    // Start curriculum study mode
    const handleStartCurriculumStudy = () => {
        setStudyMode('curriculum')
        setCurriculumModalOpen(false)
        setCurrentScreen('study')
        fetchCurriculumFlashcard()
    }

    // Clear curriculum and return to chapter mode
    const handleClearCurriculum = () => {
        setActiveCurriculum(null)
        setStudyMode('chapter')
        localStorage.removeItem('studybuddy-curriculum')
        setCurriculumModalOpen(false)
    }

    // Handlers
    const handleStartStudy = () => {
        setStudyMode('chapter')
        setCurrentScreen('study')
        fetchFlashcard()
    }

    const handleBack = () => {
        setCurrentScreen('home')
        setCurrentCard(null)
        setChatMessages([])
        setChatOpen(false)
    }

    const handleReview = async (button: 'no' | 'took_a_sec' | 'yes') => {
        // Record review for spaced repetition
        if (currentCard?.flashcard_id) {
            try {
                await fetch('/api/review', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        flashcard_id: currentCard.flashcard_id,
                        button,
                    }),
                })
                // Refresh focus areas after review
                fetchFocusAreas()
            } catch (err) {
                console.error('Failed to record review:', err)
            }
        }

        // Fetch next card based on study mode and review result
        if (studyMode === 'curriculum') {
            if (button === 'no' && currentCard) {
                fetchCurriculumFlashcard(currentCard.topic, currentCard.question)
            } else {
                fetchCurriculumFlashcard()
            }
        } else {
            if (button === 'no' && currentCard) {
                fetchFlashcard(currentCard.topic, currentCard.question)
            } else {
                fetchFlashcard()
            }
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

    // Open chat with a pre-filled message about a focus area topic
    const handleChatAboutTopic = (topic: string) => {
        setChatPrefill(`Can you help me understand "${topic}" better? I've been struggling with this concept.`)
        setChatOpen(true)
    }

    const handleSendMessage = async (message: string) => {
        // Clear prefill after sending
        setChatPrefill('')

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
                    studyMode={studyMode}
                    activeCurriculum={activeCurriculum}
                    focusAreas={focusAreas}
                    onChatAboutTopic={handleChatAboutTopic}
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
                        activeCurriculum={activeCurriculum}
                        onOpenCurriculumModal={() => setCurriculumModalOpen(true)}
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
                prefillMessage={chatPrefill}
            />

            {/* Curriculum modal */}
            <CurriculumModal
                open={curriculumModalOpen}
                onClose={() => setCurriculumModalOpen(false)}
                activeCurriculum={activeCurriculum}
                onCreateCurriculum={handleCreateCurriculum}
                onStartStudy={handleStartCurriculumStudy}
                onClearCurriculum={handleClearCurriculum}
                isCreating={isCreatingCurriculum}
            />
        </div>
    )
}
