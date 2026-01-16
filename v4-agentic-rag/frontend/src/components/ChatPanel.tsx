'use client'

import { useEffect, useRef } from 'react'
import ChatMessage from './ChatMessage'
import ChatInput from './ChatInput'

interface Message {
    id: string
    content: string
    role: 'user' | 'assistant'
    isLoading?: boolean
}

interface ChatPanelProps {
    open: boolean
    messages: Message[]
    onClose: () => void
    onSendMessage: (message: string) => void
    isLoading: boolean
}

export default function ChatPanel({
    open,
    messages,
    onClose,
    onSendMessage,
    isLoading,
}: ChatPanelProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null)

    // Scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    return (
        <>
            {/* Backdrop */}
            {open && (
                <div
                    className="fixed inset-0 bg-black/20 z-40 sm:hidden"
                    onClick={onClose}
                    aria-hidden="true"
                />
            )}

            {/* Panel */}
            <div
                className={`chat-panel z-50 flex flex-col ${open ? 'open' : ''}`}
                role="dialog"
                aria-label="Chat panel"
                aria-hidden={open ? 'false' : 'true'}
            >
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200">
                    <h2 className="text-lg font-semibold text-gray-900">Chat</h2>
                    <button
                        type="button"
                        onClick={onClose}
                        className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100"
                        aria-label="Close chat"
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-5 w-5"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                        >
                            <path
                                fillRule="evenodd"
                                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                                clipRule="evenodd"
                            />
                        </svg>
                    </button>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                    {messages.length === 0 ? (
                        <p className="text-center text-gray-500 text-sm">
                            Ask a question about the current flashcard or any topic!
                        </p>
                    ) : (
                        messages.map((msg) => (
                            <ChatMessage
                                key={msg.id}
                                content={msg.content}
                                role={msg.role}
                                isLoading={msg.isLoading}
                            />
                        ))
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="p-4 border-t border-gray-200">
                    <ChatInput onSend={onSendMessage} disabled={isLoading} />
                    <p className="text-center text-xs text-gray-400 mt-2">
                        Press Escape to close
                    </p>
                </div>
            </div>
        </>
    )
}
