'use client'

interface ReviewButtonsProps {
    onReview: (button: 'no' | 'took_a_sec' | 'yes') => void
    disabled?: boolean
}

export default function ReviewButtons({ onReview, disabled = false }: ReviewButtonsProps) {
    return (
        <div className="flex gap-3 max-w-md mx-auto">
            <button
                onClick={() => onReview('no')}
                disabled={disabled}
                className="flex-1 py-3 px-4 btn-no font-medium rounded-lg focus:ring-2 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="No, didn't remember"
            >
                No
            </button>
            <button
                onClick={() => onReview('took_a_sec')}
                disabled={disabled}
                className="flex-1 py-3 px-4 btn-took-a-sec font-medium rounded-lg focus:ring-2 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Took a moment to remember"
            >
                Took a sec
            </button>
            <button
                onClick={() => onReview('yes')}
                disabled={disabled}
                className="flex-1 py-3 px-4 btn-yes font-medium rounded-lg focus:ring-2 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Yes, remembered it"
            >
                Yes
            </button>
        </div>
    )
}
