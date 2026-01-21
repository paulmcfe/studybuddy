'use client'

import { useState } from 'react'

interface Card {
    id: string
    topic: string
    question: string
    answer: string
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

    return (
        <div>
            <div className="flashcard" onClick={handleClick}>
                <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginBottom: '1rem' }}>
                    {card.topic}
                </div>
                <div className="flashcard-question">{card.question}</div>
                {showAnswer && (
                    <div className="flashcard-answer">{card.answer}</div>
                )}
                {!showAnswer && (
                    <p style={{ marginTop: '2rem', color: 'var(--color-text-muted)', fontSize: '0.875rem' }}>
                        Click to reveal answer
                    </p>
                )}
            </div>

            <div style={{ textAlign: 'center', marginTop: '1rem' }}>
                <p style={{ marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>
                    Did you remember this?
                </p>
                <div className="flashcard-actions" style={{ justifyContent: 'center' }}>
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
                        Took a sec
                    </button>
                    <button
                        className="btn-success"
                        onClick={() => handleReview(5)}
                    >
                        Yes
                    </button>
                </div>
            </div>
        </div>
    )
}
