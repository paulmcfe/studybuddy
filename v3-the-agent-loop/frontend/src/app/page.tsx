'use client'

import { useState, useEffect } from 'react'
import MessageList, { ChatMessage } from '@/components/MessageList'
import MessageInput from '@/components/MessageInput'

interface AgentStatus {
    indexing_complete: boolean
    documents_indexed: number
    chunks_in_db: number
    current_file: string
}

const WELCOME_MESSAGE: ChatMessage = {
    id: 'welcome',
    content: "Hi! I'm StudyBuddy v3, your AI tutoring agent! I use LangChain and Qdrant to intelligently search through Sherlock Holmes stories. Try asking me something like \"Who is Irene Adler?\" or \"What happened in The Red-Headed League?\"",
    role: 'assistant',
}

export default function Home() {
    const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null)

    // Poll agent status on mount
    useEffect(() => {
        const checkStatus = async () => {
            try {
                const res = await fetch('/api/status')
                if (res.ok) {
                    const data = await res.json()
                    setAgentStatus(data)
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

            // Replace loading message with actual response + reasoning
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === loadingMessageId
                        ? { ...msg, content: data.reply, isLoading: false, reasoning: data.reasoning }
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

            {/* Version tag with agent status */}
            <div className="text-center text-xs text-gray-400 py-2 bg-white border-t border-gray-100">
                StudyBuddy v3 ·{' '}
                {agentStatus?.indexing_complete ? (
                    <span className="text-green-600">
                        {agentStatus.documents_indexed} stories indexed ({agentStatus.chunks_in_db} chunks)
                    </span>
                ) : agentStatus?.current_file ? (
                    <span className="text-amber-500">
                        Indexing &quot;{agentStatus.current_file}&quot;... ({agentStatus.chunks_in_db} chunks)
                    </span>
                ) : (
                    <span className="text-amber-500">Connecting...</span>
                )}
            </div>
        </div>
    )
}
