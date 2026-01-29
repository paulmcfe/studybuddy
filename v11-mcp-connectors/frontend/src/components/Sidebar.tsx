'use client'

import { useState, useEffect } from 'react'

interface Program {
    id: string
    name: string
    document_count: number
    flashcard_count: number
}

type View = 'overview' | 'study' | 'flashcards' | 'documents' | 'connectors' | 'new-program'

interface SidebarProps {
    programs: Program[]
    selectedProgram: Program | null
    onSelectProgram: (program: Program) => void
    currentView: View
    onNavigate: (view: View) => void
}

export default function Sidebar({
    programs,
    selectedProgram,
    onSelectProgram,
    currentView,
    onNavigate,
}: SidebarProps) {
    const [theme, setTheme] = useState<'light' | 'dark'>('light')

    useEffect(() => {
        // Read initial theme from document
        const currentTheme = document.documentElement.getAttribute('data-theme') as 'light' | 'dark'
        if (currentTheme) {
            setTheme(currentTheme)
        }
    }, [])

    const toggleTheme = () => {
        const newTheme = theme === 'dark' ? 'light' : 'dark'
        setTheme(newTheme)
        document.documentElement.setAttribute('data-theme', newTheme)
        localStorage.setItem('theme', newTheme)
    }

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="sidebar-brand">
                    <img
                        src="/images/studybuddy-icon.png"
                        alt=""
                        className="sidebar-brand-icon"
                    />
                    <h1 className="sidebar-brand-title">StudyBuddy</h1>
                </div>
                <span className="sidebar-version">v11 - MCP Connectors</span>
            </div>

            <div className="sidebar-section">
                <div className="flex justify-between items-center">
                    <h3 className="sidebar-section-title">Programs</h3>
                    <button
                        onClick={() => onNavigate('new-program')}
                        className="btn-secondary btn-sm"
                    >
                        + New
                    </button>
                </div>

                <ul className="program-list">
                    {programs.length === 0 ? (
                        <li className="empty-state">No programs yet</li>
                    ) : (
                        programs.map((program) => (
                            <li
                                key={program.id}
                                className={`program-item ${
                                    selectedProgram?.id === program.id ? 'active' : ''
                                }`}
                                onClick={() => onSelectProgram(program)}
                            >
                                <div className="program-name">{program.name}</div>
                                <div className="program-meta">
                                    {program.document_count} {program.document_count === 1 ? 'doc' : 'docs'} · {program.flashcard_count} {program.flashcard_count === 1 ? 'card' : 'cards'}
                                </div>
                            </li>
                        ))
                    )}
                </ul>
            </div>

            {selectedProgram && (
                <nav className="sidebar-nav">
                    <h3 className="sidebar-nav-title">{selectedProgram.name}</h3>
                    <button
                        onClick={() => onNavigate('overview')}
                        className={`sidebar-nav-btn tab ${currentView === 'overview' ? 'active' : ''}`}
                    >
                        Overview
                    </button>
                    <button
                        onClick={() => onNavigate('flashcards')}
                        className={`sidebar-nav-btn tab ${currentView === 'study' || currentView === 'flashcards' ? 'active' : ''}`}
                    >
                        Study
                    </button>
                    <button
                        onClick={() => onNavigate('documents')}
                        className={`sidebar-nav-btn tab ${currentView === 'documents' ? 'active' : ''}`}
                    >
                        Documents
                    </button>
                    <button
                        onClick={() => onNavigate('connectors')}
                        className={`sidebar-nav-btn tab ${currentView === 'connectors' ? 'active' : ''}`}
                    >
                        Connectors
                    </button>
                </nav>
            )}

            <div className="sidebar-footer">
                <button
                    type="button"
                    onClick={toggleTheme}
                    className="theme-toggle mb-sm"
                    aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                >
                    {theme === 'dark' ? '☀️' : '🌙'}
                </button>
                <a href="/api/health" target="_blank" className="sidebar-link">
                    API Health Check
                </a>
            </div>
        </aside>
    )
}
