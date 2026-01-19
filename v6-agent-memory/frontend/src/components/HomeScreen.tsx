'use client'

import { useEffect, useState } from 'react'

interface Chapter {
    id: number
    title: string
    sections: { name: string; subtopics: string[] }[]
}

interface HomeScreenProps {
    chapters: Chapter[]
    selectedChapter: number | null
    scope: 'single' | 'cumulative'
    onChapterChange: (chapterId: number) => void
    onScopeChange: (scope: 'single' | 'cumulative') => void
    onStart: () => void
    disabled: boolean
    statusMessage: string
}

export default function HomeScreen({
    chapters,
    selectedChapter,
    scope,
    onChapterChange,
    onScopeChange,
    onStart,
    disabled,
    statusMessage,
}: HomeScreenProps) {
    const showScopeToggle = selectedChapter !== null && selectedChapter > 1
    const [theme, setTheme] = useState<'light' | 'dark'>('light')

    useEffect(() => {
        // Read initial theme from document
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
        <div className="flex flex-col items-center justify-center min-h-[80vh] px-4">
            {/* Theme toggle in top right */}
            <button
                type="button"
                onClick={toggleTheme}
                className="theme-toggle fixed top-4 right-4"
                aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
                {theme === 'dark' ? '☀️' : '🌙'}
            </button>

            <div className="w-full max-w-md space-y-6">
                <div className="text-center">
                    <div className="flex items-center justify-center gap-3 mb-2">
                        <img
                            src="/images/studybuddy-icon.png"
                            alt=""
                            className="w-10 h-10"
                        />
                        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">StudyBuddy</h1>
                    </div>
                    <p className="text-gray-600 dark:text-gray-400">AI Engineering Flashcards</p>
                </div>

                <div className="space-y-4">
                    <div>
                        <label
                            htmlFor="chapter-select"
                            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                        >
                            Select Chapter
                        </label>
                        <select
                            id="chapter-select"
                            value={selectedChapter ?? ''}
                            onChange={(e) => onChapterChange(Number(e.target.value))}
                            disabled={chapters.length === 0}
                            className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:text-gray-500"
                            aria-describedby="chapter-status"
                        >
                            <option value="">
                                {chapters.length === 0 ? 'Loading chapters...' : 'Choose a chapter'}
                            </option>
                            {chapters.map((chapter) => (
                                <option key={chapter.id} value={chapter.id}>
                                    Chapter {chapter.id}: {chapter.title}
                                </option>
                            ))}
                        </select>
                    </div>

                    {showScopeToggle && (
                        <div
                            className="flex gap-4"
                            role="radiogroup"
                            aria-label="Study scope"
                        >
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="radio"
                                    name="scope"
                                    value="single"
                                    checked={scope === 'single'}
                                    onChange={() => onScopeChange('single')}
                                    className="w-4 h-4 text-blue-600"
                                />
                                <span className="text-gray-700 dark:text-gray-300">Single chapter</span>
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="radio"
                                    name="scope"
                                    value="cumulative"
                                    checked={scope === 'cumulative'}
                                    onChange={() => onScopeChange('cumulative')}
                                    className="w-4 h-4 text-blue-600"
                                />
                                <span className="text-gray-700 dark:text-gray-300">
                                    Cumulative (1-{selectedChapter})
                                </span>
                            </label>
                        </div>
                    )}

                    <button
                        onClick={onStart}
                        disabled={disabled || selectedChapter === null}
                        className="w-full py-3 px-4 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                        aria-describedby="start-status"
                    >
                        Start Studying
                    </button>

                    <p
                        id="chapter-status"
                        className="text-center text-sm text-gray-500 dark:text-gray-300"
                        aria-live="polite"
                    >
                        {statusMessage}
                    </p>
                </div>
            </div>
        </div>
    )
}
