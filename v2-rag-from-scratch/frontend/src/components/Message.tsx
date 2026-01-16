import ReactMarkdown from 'react-markdown'
import LoadingDots from './LoadingDots'

export type MessageRole = 'user' | 'assistant' | 'error'

interface MessageProps {
    content: string
    role: MessageRole
    isLoading?: boolean
}

export default function Message({ content, role, isLoading }: MessageProps) {
    const isUser = role === 'user'
    const isError = role === 'error'

    return (
        <div
            className={`message-animate flex ${isUser ? 'justify-end' : 'justify-start'}`}
        >
            <div
                className={`
                    max-w-[80%] px-4 py-3 rounded-2xl shadow-sm
                    ${isUser
                        ? 'bg-[var(--color-user-bg)] text-white rounded-br-md'
                        : isError
                            ? 'bg-[var(--color-error-bg)] text-[var(--color-error-text)] rounded-bl-md'
                            : 'bg-[var(--color-assistant-bg)] text-gray-800 rounded-bl-md'
                    }
                `}
            >
                {isLoading ? (
                    <LoadingDots />
                ) : (
                    <div className={`markdown-content ${isUser ? 'user-markdown' : ''}`}>
                        <ReactMarkdown>{content}</ReactMarkdown>
                    </div>
                )}
            </div>
        </div>
    )
}
