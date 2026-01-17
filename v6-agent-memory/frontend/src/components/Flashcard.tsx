'use client'

interface FlashcardData {
    question: string
    answer: string
    topic: string
    source: 'rag' | 'llm'
}

interface FlashcardProps {
    card: FlashcardData
    isFlipped: boolean
    onFlip: () => void
}

export default function Flashcard({ card, isFlipped, onFlip }: FlashcardProps) {
    return (
        <div className="flashcard-container mx-auto">
            <div
                className={`flashcard ${isFlipped ? 'flipped' : ''}`}
                onClick={onFlip}
                onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        e.stopPropagation()
                        onFlip()
                    }
                }}
                role="button"
                tabIndex={0}
                aria-label={isFlipped ? 'Flashcard showing answer. Click to see question.' : 'Flashcard showing question. Click to see answer.'}
            >
                {/* Front - Question */}
                <div className="flashcard-face flashcard-front">
                    <div className="topic-badge">{card.topic}</div>
                    <div className="flex-1 flex items-center justify-center">
                        <p className="text-lg text-gray-800 dark:text-gray-100 text-center">{card.question}</p>
                    </div>
                    <div className="text-center text-sm text-gray-400">
                        Tap to reveal answer
                    </div>
                </div>

                {/* Back - Answer */}
                <div className="flashcard-face flashcard-back">
                    <div className="topic-badge">{card.topic}</div>
                    <div className="flex-1 flex items-center justify-center overflow-auto">
                        <p className="text-base text-gray-700 dark:text-gray-100 text-center">{card.answer}</p>
                    </div>
                    <div className="flex justify-center">
                        <span className={`source-badge ${card.source === 'rag' ? 'source-rag' : 'source-llm'}`}>
                            {card.source === 'rag' ? 'From study materials' : 'Based on AI knowledge'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    )
}
