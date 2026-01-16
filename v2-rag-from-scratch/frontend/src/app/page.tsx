'use client'

import { useState, useEffect } from 'react'
import MessageList, { ChatMessage } from '@/components/MessageList'
import MessageInput from '@/components/MessageInput'

interface RagStatus {
    indexing_complete: boolean
    chunks_in_db: number
}

const WELCOME_MESSAGE: ChatMessage = {
    id: 'welcome',
    content: "Hi! I'm StudyBuddy with RAG! I have access to study materials about **mitosis**, **photosynthesis**, and the **water cycle**. Ask me anything about these topics!",
    role: 'assistant',
}

export default function Home() {
    const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [ragStatus, setRagStatus] = useState<RagStatus | null>(null)

    // Poll RAG status on mount
    useEffect(() => {
        const checkStatus = async () => {
            try {
                const res = await fetch('/api/status')
                if (res.ok) {
                    const data = await res.json()
                    setRagStatus(data)
                }
            } catch {
                // Retry on failure - backend might not be ready yet
            }
        }

        checkStatus()
        const interval = setInterval(checkStatus, 2000)

        return () => clearInterval(interval)
    }, [])

    const handleSend = async () => {
        const userMessage = input.trim()
        if (!userMessage || isLoading) return

        // Add user message
        const userMessageObj: ChatMessage = {
            id: `user-${Date.now()}`,
            content: userMessage,
            role: 'user',
        }

        // Add loading message
        const loadingMessageId = `assistant-${Date.now()}`
        const loadingMessage: ChatMessage = {
            id: loadingMessageId,
            content: '',
            role: 'assistant',
            isLoading: true,
        }

        setMessages((prev) => [...prev, userMessageObj, loadingMessage])
        setInput('')
        setIsLoading(true)

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage }),
            })

            if (!response.ok) {
                throw new Error(
                    response.status === 500
                        ? 'Server error. Please check that the API key is configured.'
                        : 'Unable to connect to the server. Please try again.'
                )
            }

            const data = await response.json()

            // Replace loading message with actual response
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === loadingMessageId
                        ? { ...msg, content: data.reply, isLoading: false }
                        : msg
                )
            )
        } catch (error) {
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : 'Something went wrong. Please try again.'

            // Replace loading message with error
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === loadingMessageId
                        ? { ...msg, content: errorMessage, role: 'error', isLoading: false }
                        : msg
                )
            )
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="flex flex-col h-screen">
            {/* Header */}
            <header className="border-b border-gray-200 bg-white px-4 py-3">
                <div className="max-w-3xl mx-auto">
                    <h1 className="text-xl font-semibold text-[var(--color-primary)]">
                        StudyBuddy
                    </h1>
                </div>
            </header>

            {/* Messages */}
            <MessageList messages={messages} />

            {/* Input */}
            <MessageInput
                value={input}
                onChange={setInput}
                onSend={handleSend}
                disabled={isLoading}
            />

            {/* Version tag with RAG status */}
            <div className="text-center text-xs text-gray-400 py-2 bg-white border-t border-gray-100">
                StudyBuddy v2 ·{' '}
                {ragStatus?.indexing_complete ? (
                    <span className="text-green-600">RAG enabled</span>
                ) : (
                    <span className="text-amber-500">Indexing...</span>
                )}
                {ragStatus && ragStatus.chunks_in_db > 0 && (
                    <span> · {ragStatus.chunks_in_db} chunks</span>
                )}
            </div>
        </div>
    )
}
