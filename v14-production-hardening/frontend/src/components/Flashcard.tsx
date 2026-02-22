'use client'

import { useState } from 'react'

interface Card {
    id: string
    topic: string
    question: string
    answer: string
    interval: number
    repetitions: number
}

function getCardStatus(card: Card): { label: string; status: string } {
    const interval = card.interval ?? 0
    const repetitions = card.repetitions ?? 0

    if (interval > 21) {
        return { label: 'Mastered', status: 'mastered' }
    } else if (repetitions === 0) {
        return { label: 'New', status: 'new' }
    } else if (interval <= 1) {
        return { label: 'Still Learning', status: 'learning' }
    } else {
        return { label: 'Reviewing', status: 'reviewing' }
    }
}

interface FlashcardProps {
    card: Card | null
    onReview: (quality: number) => void
}

export default function Flashcard({ card, onReview }: FlashcardProps) {
    const [showAnswer, setShowAnswer] = useState(false)

    if (!card) return null

    const handleClick = () => {
        if (!showAnswer) {
            setShowAnswer(true)
        }
    }

    const handleReview = (quality: number) => {
        setShowAnswer(false)
        onReview(quality)
    }

    const status = getCardStatus(card)

    return (
        <div>
            <div className="flashcard" onClick={handleClick}>
                <div className="flashcard-header">
                    <span className="flashcard-topic">
                        {card.topic}
                    </span>
                    <span className="flashcard-status" data-status={status.status}>
                        {status.label}
                    </span>
                </div>
                <div className="flashcard-question">{card.question}</div>
                {showAnswer && (
                    <div className="flashcard-answer">{card.answer}</div>
                )}
                {!showAnswer && (
                    <p className="flashcard-hint">
                        Click to reveal answer
                    </p>
                )}
            </div>

            <div className="flashcard-review">
                <p className="flashcard-review-question">
                    Did you know it?
                </p>
                <div className="flashcard-actions flashcard-actions-centered">
                    <button
                        className="btn-danger"
                        onClick={() => handleReview(1)}
                    >
                        No
                    </button>
                    <button
                        className="btn-secondary"
                        onClick={() => handleReview(3)}
                    >
                        Struggled
                    </button>
                    <button
                        className="btn-success"
                        onClick={() => handleReview(5)}
                    >
                        Yes
                    </button>
                </div>
                <p className="flashcard-review-count">
                    {(card.repetitions ?? 0) === 0
                        ? 'First time seeing this card'
                        : `Reviewed ${card.repetitions} time${card.repetitions === 1 ? '' : 's'}`}
                </p>
            </div>
        </div>
    )
}
