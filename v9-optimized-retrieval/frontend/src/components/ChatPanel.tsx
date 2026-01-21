'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import ChatMessage from './ChatMessage'
import ChatInput from './ChatInput'

const MIN_WIDTH = 300
const MAX_WIDTH = 700
const DEFAULT_WIDTH = 400

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
    prefillMessage?: string
}

export default function ChatPanel({
    open,
    messages,
    onClose,
    onSendMessage,
    isLoading,
    prefillMessage,
}: ChatPanelProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const [width, setWidth] = useState(DEFAULT_WIDTH)
    const [isResizing, setIsResizing] = useState(false)

    // Load saved width from localStorage
    useEffect(() => {
        const saved = localStorage.getItem('chat-panel-width')
        if (saved) {
            const parsed = parseInt(saved, 10)
            if (parsed >= MIN_WIDTH && parsed <= MAX_WIDTH) {
                setWidth(parsed)
            }
        }
    }, [])

    // Save width to localStorage when it changes
    useEffect(() => {
        if (width !== DEFAULT_WIDTH) {
            localStorage.setItem('chat-panel-width', String(width))
        }
    }, [width])

    // Handle resize drag
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        e.preventDefault()
        setIsResizing(true)
    }, [])

    useEffect(() => {
        if (!isResizing) return

        const handleMouseMove = (e: MouseEvent) => {
            const newWidth = window.innerWidth - e.clientX
            setWidth(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, newWidth)))
        }

        const handleMouseUp = () => {
            setIsResizing(false)
        }

        document.addEventListener('mousemove', handleMouseMove)
        document.addEventListener('mouseup', handleMouseUp)

        return () => {
            document.removeEventListener('mousemove', handleMouseMove)
            document.removeEventListener('mouseup', handleMouseUp)
        }
    }, [isResizing])

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
                className={`chat-panel z-50 flex flex-col ${open ? 'open' : ''} ${isResizing ? 'select-none' : ''}`}
                style={{ '--chat-panel-width': `${width}px` } as React.CSSProperties}
                role="dialog"
                aria-label="Chat panel"
                aria-hidden={!open}
                inert={!open ? true : undefined}
            >
                {/* Resize handle - desktop only */}
                <div
                    className="hidden sm:block absolute left-0 top-0 bottom-0 w-1 cursor-ew-resize hover:bg-blue-500/50 active:bg-blue-500/50 transition-colors"
                    onMouseDown={handleMouseDown}
                    aria-hidden="true"
                />

                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Chat</h2>
                    <button
                        type="button"
                        onClick={onClose}
                        className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
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
                <div className="p-4 border-t border-gray-200 dark:border-gray-700">
                    <ChatInput onSend={onSendMessage} disabled={isLoading} defaultValue={prefillMessage} />
                    <p className="text-center text-xs text-gray-400 dark:text-gray-500 mt-2">
                        Press Escape to close
                    </p>
                </div>
            </div>
        </>
    )
}
