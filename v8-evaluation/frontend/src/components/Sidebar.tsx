'use client'

import FocusAreas from './FocusAreas'

interface Chapter {
    id: number
    title: string
    sections: { name: string; subtopics: string[] }[]
}

interface AgentStatus {
    indexing_complete: boolean
    documents_indexed: number
    chunks_in_db: number
    current_file: string
}

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

interface SidebarProps {
    chapter: Chapter | null
    scope: 'single' | 'cumulative'
    agentStatus: AgentStatus | null
    studyMode: 'chapter' | 'curriculum'
    activeCurriculum: Curriculum | null
    focusAreas: string[]
    onChatAboutTopic: (topic: string) => void
}

export default function Sidebar({
    chapter,
    scope,
    agentStatus,
    studyMode,
    activeCurriculum,
    focusAreas,
    onChatAboutTopic,
}: SidebarProps) {
    return (
        <aside className="hidden sm:flex flex-col w-64 border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 p-4">
            <div className="space-y-6">
                {/* Study Info */}
                <div>
                    <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                        Currently Studying
                    </h2>
                    {studyMode === 'curriculum' && activeCurriculum ? (
                        <div className="space-y-1">
                            <p className="font-medium text-gray-900 dark:text-gray-100">
                                Learning Path
                            </p>
                            <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-2">
                                {activeCurriculum.goal}
                            </p>
                            {activeCurriculum.currentModule && (
                                <p className="text-xs text-gray-500 dark:text-gray-400">
                                    Module: {activeCurriculum.currentModule.title}
                                </p>
                            )}
                        </div>
                    ) : chapter ? (
                        <div className="space-y-1">
                            <p className="font-medium text-gray-900 dark:text-gray-100">
                                Chapter {chapter.id}
                            </p>
                            <p className="text-sm text-gray-600 dark:text-gray-300">{chapter.title}</p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                                {scope === 'cumulative'
                                    ? `Cumulative (Chapters 1-${chapter.id})`
                                    : 'Single chapter'}
                            </p>
                        </div>
                    ) : (
                        <p className="text-sm text-gray-500 dark:text-gray-400">No chapter selected</p>
                    )}
                </div>

                {/* Agent Status */}
                <div>
                    <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                        Agent Status
                    </h2>
                    {agentStatus ? (
                        <div className="space-y-1">
                            <div className="flex items-center gap-2">
                                <div
                                    className={`w-2 h-2 rounded-full ${
                                        agentStatus.indexing_complete
                                            ? 'bg-green-500'
                                            : 'bg-amber-500 animate-pulse'
                                    }`}
                                />
                                <span className="text-sm text-gray-600 dark:text-gray-300">
                                    {agentStatus.indexing_complete ? 'Ready' : 'Indexing...'}
                                </span>
                            </div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                                {agentStatus.documents_indexed} guides indexed
                            </p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                                {agentStatus.chunks_in_db} chunks
                            </p>
                        </div>
                    ) : (
                        <p className="text-sm text-gray-500 dark:text-gray-400">Connecting...</p>
                    )}
                </div>

                {/* Focus Areas */}
                {focusAreas.length > 0 && (
                    <FocusAreas areas={focusAreas} onChatAboutTopic={onChatAboutTopic} />
                )}

                {/* Keyboard shortcuts */}
                <div>
                    <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                        Shortcuts
                    </h2>
                    <div className="space-y-1 text-xs text-gray-500 dark:text-gray-400">
                        <p>
                            <kbd className="px-1 py-0.5 bg-gray-200 dark:bg-gray-600 rounded">Space</kbd>{' '}
                            Flip card
                        </p>
                        <p>
                            <kbd className="px-1 py-0.5 bg-gray-200 dark:bg-gray-600 rounded">Esc</kbd>{' '}
                            Close chat
                        </p>
                    </div>
                </div>
            </div>
        </aside>
    )
}
