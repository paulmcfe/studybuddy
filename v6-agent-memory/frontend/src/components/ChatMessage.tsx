'use client'

import ReactMarkdown from 'react-markdown'
import LoadingDots from './LoadingDots'

interface ChatMessageProps {
    content: string
    role: 'user' | 'assistant'
    isLoading?: boolean
}

export default function ChatMessage({ content, role, isLoading }: ChatMessageProps) {
    const isUser = role === 'user'

    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
            <div
                className={`max-w-[85%] px-4 py-2 rounded-2xl ${
                    isUser
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100'
                }`}
            >
                {isLoading ? (
                    <LoadingDots />
                ) : isUser ? (
                    <p className="text-sm">{content}</p>
                ) : (
                    <div className="markdown-content text-sm">
                        <ReactMarkdown>{content}</ReactMarkdown>
                    </div>
                )}
            </div>
        </div>
    )
}
