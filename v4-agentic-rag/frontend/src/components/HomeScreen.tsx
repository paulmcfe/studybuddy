'use client'

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

    return (
        <div className="flex flex-col items-center justify-center min-h-[80vh] px-4">
            <div className="w-full max-w-md space-y-6">
                <div className="text-center">
                    <h1 className="text-3xl font-bold text-gray-900 mb-2">StudyBuddy</h1>
                    <p className="text-gray-600">AI Engineering Flashcards</p>
                </div>

                <div className="space-y-4">
                    <div>
                        <label
                            htmlFor="chapter-select"
                            className="block text-sm font-medium text-gray-700 mb-1"
                        >
                            Select Chapter
                        </label>
                        <select
                            id="chapter-select"
                            value={selectedChapter ?? ''}
                            onChange={(e) => onChapterChange(Number(e.target.value))}
                            disabled={chapters.length === 0}
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg text-gray-900 bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
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
                                <span className="text-gray-700">Single chapter</span>
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
                                <span className="text-gray-700">
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
                        className="text-center text-sm text-gray-500"
                        aria-live="polite"
                    >
                        {statusMessage}
                    </p>
                </div>
            </div>
        </div>
    )
}
