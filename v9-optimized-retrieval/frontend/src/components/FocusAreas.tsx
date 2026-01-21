'use client'

interface FocusAreasProps {
    areas: string[]
    onChatAboutTopic: (topic: string) => void
}

export default function FocusAreas({ areas, onChatAboutTopic }: FocusAreasProps) {
    if (areas.length === 0) {
        return null
    }

    return (
        <div className="focus-areas">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Focus Areas
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                Topics where you need extra practice
            </p>
            <ul className="space-y-2">
                {areas.map((topic, index) => (
                    <li
                        key={index}
                        className="text-sm text-gray-600 dark:text-gray-300 flex items-start justify-between gap-2"
                    >
                        <span className="flex-1">{topic}</span>
                        <button
                            onClick={() => onChatAboutTopic(topic)}
                            className="text-xs text-blue-600 dark:text-blue-400 hover:underline whitespace-nowrap"
                        >
                            Chat about this
                        </button>
                    </li>
                ))}
            </ul>
        </div>
    )
}
