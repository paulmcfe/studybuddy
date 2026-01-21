'use client'

import { useState } from 'react'
import LoadingDots from './LoadingDots'

interface CurriculumModule {
    id: string
    title: string
    description?: string
    estimated_hours?: number
    topics?: string[]
    status?: 'pending' | 'in_progress' | 'completed'
}

interface Curriculum {
    id: string
    goal: string
    modules: CurriculumModule[]
    currentModule: CurriculumModule | null
}

interface CurriculumModalProps {
    open: boolean
    onClose: () => void
    activeCurriculum: Curriculum | null
    onCreateCurriculum: (goal: string, weeklyHours: number) => Promise<Curriculum | null>
    onStartStudy: () => void
    onClearCurriculum: () => void
    isCreating: boolean
}

export default function CurriculumModal({
    open,
    onClose,
    activeCurriculum,
    onCreateCurriculum,
    onStartStudy,
    onClearCurriculum,
    isCreating,
}: CurriculumModalProps) {
    const [goal, setGoal] = useState('')
    const [weeklyHours, setWeeklyHours] = useState(5)

    if (!open) return null

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!goal.trim()) return
        await onCreateCurriculum(goal.trim(), weeklyHours)
    }

    const handleClear = () => {
        setGoal('')
        setWeeklyHours(5)
        onClearCurriculum()
    }

    const getModuleIcon = (status?: string) => {
        switch (status) {
            case 'completed':
                return '✓'
            case 'in_progress':
                return '→'
            default:
                return '○'
        }
    }

    return (
        <div
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={onClose}
        >
            <div
                className="bg-white dark:bg-[hsl(220,15%,18%)] rounded-xl shadow-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                        {activeCurriculum ? 'Your Learning Path' : 'Create Learning Path'}
                    </h2>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-xl"
                        aria-label="Close"
                    >
                        ×
                    </button>
                </div>

                {/* Content */}
                <div className="p-4">
                    {isCreating ? (
                        /* Loading State */
                        <div className="py-12 text-center">
                            <LoadingDots />
                            <p className="mt-4 text-gray-600 dark:text-gray-400">
                                Creating your personalized learning path...
                            </p>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-500">
                                This may take a moment
                            </p>
                        </div>
                    ) : activeCurriculum ? (
                        /* Curriculum Result View */
                        <div>
                            <div className="mb-4">
                                <p className="text-sm text-gray-500 dark:text-gray-400">
                                    Your Goal
                                </p>
                                <p className="text-gray-900 dark:text-white font-medium">
                                    {activeCurriculum.goal}
                                </p>
                            </div>

                            <div className="mb-4">
                                <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                                    Learning Modules
                                </p>
                                <ul className="space-y-2">
                                    {activeCurriculum.modules.map((module, index) => (
                                        <li
                                            key={module.id || index}
                                            className="curriculum-module-card flex items-start gap-3 p-3 rounded-lg bg-gray-50 dark:bg-[hsl(220,15%,15%)]"
                                        >
                                            <span
                                                className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                                                    module.status === 'completed'
                                                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                                        : module.status === 'in_progress'
                                                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                                                        : 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                                                }`}
                                            >
                                                {getModuleIcon(module.status)}
                                            </span>
                                            <div className="flex-1 min-w-0">
                                                <p className="font-medium text-gray-900 dark:text-white">
                                                    {module.title}
                                                </p>
                                                {module.description && (
                                                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                                                        {module.description}
                                                    </p>
                                                )}
                                                {module.estimated_hours && (
                                                    <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                                                        ~{module.estimated_hours} hours
                                                    </p>
                                                )}
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            <div className="flex gap-3 mt-6">
                                <button
                                    onClick={onStartStudy}
                                    className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
                                >
                                    {goal.trim() ? 'Start Learning' : 'Resume Learning'}
                                </button>
                                <button
                                    onClick={handleClear}
                                    className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
                                >
                                    Clear
                                </button>
                            </div>
                        </div>
                    ) : (
                        /* Form View */
                        <form onSubmit={handleSubmit}>
                            <div className="mb-4">
                                <label
                                    htmlFor="goal"
                                    className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                                >
                                    What do you want to learn?
                                </label>
                                <textarea
                                    id="goal"
                                    value={goal}
                                    onChange={(e) => setGoal(e.target.value)}
                                    placeholder="e.g., I want to master building RAG applications with LangChain"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-[hsl(220,15%,15%)] text-gray-900 dark:text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                                    rows={3}
                                    required
                                />
                            </div>

                            <div className="mb-6">
                                <label
                                    htmlFor="weeklyHours"
                                    className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                                >
                                    Hours per week you can dedicate
                                </label>
                                <select
                                    id="weeklyHours"
                                    value={weeklyHours}
                                    onChange={(e) => setWeeklyHours(Number(e.target.value))}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-[hsl(220,15%,15%)] text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                >
                                    <option value={2}>2 hours/week</option>
                                    <option value={5}>5 hours/week</option>
                                    <option value={10}>10 hours/week</option>
                                    <option value={20}>20 hours/week</option>
                                </select>
                            </div>

                            <button
                                type="submit"
                                disabled={!goal.trim()}
                                className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
                            >
                                Generate Learning Path
                            </button>
                        </form>
                    )}
                </div>
            </div>
        </div>
    )
}
