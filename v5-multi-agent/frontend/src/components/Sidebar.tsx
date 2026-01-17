'use client'

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

interface SidebarProps {
    chapter: Chapter | null
    scope: 'single' | 'cumulative'
    agentStatus: AgentStatus | null
}

export default function Sidebar({ chapter, scope, agentStatus }: SidebarProps) {
    return (
        <aside className="hidden sm:flex flex-col w-64 border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 p-4">
            <div className="space-y-6">
                {/* Study Info */}
                <div>
                    <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                        Currently Studying
                    </h2>
                    {chapter ? (
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
