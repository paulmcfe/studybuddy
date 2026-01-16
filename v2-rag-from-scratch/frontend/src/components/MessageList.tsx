'use client'

import { useEffect, useRef } from 'react'
import Message, { MessageRole } from './Message'

export interface ChatMessage {
    id: string
    content: string
    role: MessageRole
    isLoading?: boolean
}

interface MessageListProps {
    messages: ChatMessage[]
}

export default function MessageList({ messages }: MessageListProps) {
    const containerRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight
        }
    }, [messages])

    return (
        <div
            ref={containerRef}
            className="flex-1 overflow-y-auto p-4 space-y-4"
        >
            <div className="max-w-3xl mx-auto space-y-4">
                {messages.map((message) => (
                    <Message
                        key={message.id}
                        content={message.content}
                        role={message.role}
                        isLoading={message.isLoading}
                    />
                ))}
            </div>
        </div>
    )
}
