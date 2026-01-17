'use client'

import { useEffect, useState } from 'react'
import Flashcard from './Flashcard'
import LoadingDots from './LoadingDots'
import ReviewButtons from './ReviewButtons'

interface FlashcardData {
    question: string
    answer: string
    topic: string
    source: 'rag' | 'llm'
}

interface StudyScreenProps {
    card: FlashcardData | null
    isFlipped: boolean
    isLoading: boolean
    onFlip: () => void
    onReview: (button: 'no' | 'took_a_sec' | 'yes') => void
    onOpenChat: () => void
    onBack: () => void
}

export default function StudyScreen({
    card,
    isFlipped,
    isLoading,
    onFlip,
    onReview,
    onOpenChat,
    onBack,
}: StudyScreenProps) {
    const [theme, setTheme] = useState<'light' | 'dark'>('light')

    useEffect(() => {
        const currentTheme = document.documentElement.getAttribute('data-theme') as 'light' | 'dark'
        if (currentTheme) {
            setTheme(currentTheme)
        }
    }, [])

    const toggleTheme = () => {
        const newTheme = theme === 'dark' ? 'light' : 'dark'
        setTheme(newTheme)
        document.documentElement.setAttribute('data-theme', newTheme)
        localStorage.setItem('theme', newTheme)
    }

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
                <button
                    onClick={onBack}
                    className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 flex items-center gap-1 py-2 px-3 -ml-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                    aria-label="Back to chapter selection"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-5 w-5"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                    >
                        <path
                            fillRule="evenodd"
                            d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z"
                            clipRule="evenodd"
                        />
                    </svg>
                    <span>Back</span>
                </button>
                <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Study Mode</h1>
                <button
                    type="button"
                    onClick={toggleTheme}
                    className="theme-toggle"
                    aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                >
                    {theme === 'dark' ? '☀️' : '🌙'}
                </button>
            </div>

            {/* Main content */}
            <div className="flex-1 flex flex-col items-center justify-center p-4">
                {isLoading ? (
                    <div className="flex flex-col items-center gap-4">
                        <LoadingDots />
                        <p className="text-gray-500 dark:text-gray-400">Generating flashcard...</p>
                    </div>
                ) : card ? (
                    <Flashcard card={card} isFlipped={isFlipped} onFlip={onFlip} />
                ) : (
                    <p className="text-gray-500 dark:text-gray-400">No flashcard loaded</p>
                )}
            </div>

            {/* Review prompt */}
            {card && !isLoading && (
                <div className="text-center text-sm font-medium text-gray-600 dark:text-gray-400 py-2">
                    Did you remember this?
                </div>
            )}

            {/* Action buttons */}
            {card && !isLoading && (
                <div className="p-4 border-t border-gray-200 dark:border-gray-700">
                    <ReviewButtons onReview={onReview} disabled={isLoading} />
                    <p className="text-center text-xs text-gray-500 dark:text-gray-400 mt-2">
                        Press Space to flip card
                    </p>
                </div>
            )}

            {/* Chat toggle button */}
            <button
                onClick={onOpenChat}
                className="fixed bottom-20 right-4 sm:bottom-4 w-14 h-14 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors flex items-center justify-center"
                aria-label="Open chat"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                    />
                </svg>
            </button>
        </div>
    )
}
