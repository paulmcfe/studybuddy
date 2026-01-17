'use client'

import ReactMarkdown from 'react-markdown'
import LoadingDots from './LoadingDots'

export interface CardData {
    question: string
    answer: string
    topic: string
    difficulty?: 'basic' | 'intermediate' | 'advanced'
    source?: string
}

interface ChatMessageProps {
    content: string
    role: 'user' | 'assistant'
    isLoading?: boolean
    cards?: CardData[]
}

export default function ChatMessage({ content, role, isLoading, cards }: ChatMessageProps) {
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
                    <>
                        <div className="markdown-content text-sm">
                            <ReactMarkdown>{content}</ReactMarkdown>
                        </div>
                        {cards && cards.length > 0 && (
                            <div className="mt-3 space-y-2 border-t border-gray-200 dark:border-gray-600 pt-3">
                                <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                                    Generated flashcards:
                                </p>
                                {cards.map((card, i) => (
                                    <div
                                        key={i}
                                        className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-600 text-xs"
                                    >
                                        <div className="flex items-center gap-2 mb-2">
                                            {card.difficulty && (
                                                <span
                                                    className={`difficulty-badge difficulty-${card.difficulty}`}
                                                >
                                                    {card.difficulty}
                                                </span>
                                            )}
                                            <span className="text-gray-500 dark:text-gray-400">
                                                {card.topic}
                                            </span>
                                        </div>
                                        <p className="font-medium text-gray-900 dark:text-gray-100 mb-1">
                                            Q: {card.question}
                                        </p>
                                        <p className="text-gray-600 dark:text-gray-300">
                                            A: {card.answer}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}
