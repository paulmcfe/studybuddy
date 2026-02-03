'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '../components/AuthContext'
import LoginPage from '../components/LoginPage'
import RegisterPage from '../components/RegisterPage'
import Sidebar from '../components/Sidebar'
import Dashboard from '../components/Dashboard'
import StudyInterface from '../components/StudyInterface'
import DocumentManager from '../components/DocumentManager'
import ConnectorManager from '../components/ConnectorManager'
import CreateProgram from '../components/CreateProgram'
import { apiFetch } from '../lib/api'

interface Program {
    id: string
    name: string
    description?: string
    document_count: number
    flashcard_count: number
}

type View = 'overview' | 'study' | 'flashcards' | 'documents' | 'connectors' | 'new-program'

export default function Home() {
    const { isAuthenticated } = useAuth()
    const [authView, setAuthView] = useState<'login' | 'register'>('login')
    const [programs, setPrograms] = useState<Program[]>([])
    const [selectedProgram, setSelectedProgram] = useState<Program | null>(null)
    const [loading, setLoading] = useState(true)
    const [view, setView] = useState<View>('overview')

    useEffect(() => {
        if (isAuthenticated) {
            loadPrograms()
        } else {
            setAuthView('login')
            setLoading(false)
        }
    }, [isAuthenticated])

    const loadPrograms = async () => {
        try {
            const response = await apiFetch('/api/programs')
            const data = await response.json()
            setPrograms(data.programs)

            if (data.programs.length > 0 && !selectedProgram) {
                setSelectedProgram(data.programs[0])
            }
        } catch (error) {
            console.error('Failed to load programs:', error)
        } finally {
            setLoading(false)
        }
    }

    // Show auth pages if not authenticated
    if (!isAuthenticated) {
        return authView === 'login'
            ? <LoginPage onSwitchToRegister={() => setAuthView('register')} />
            : <RegisterPage onSwitchToLogin={() => setAuthView('login')} />
    }

    const handleProgramCreated = (program: Program, navigateTo: 'overview' | 'documents' = 'overview') => {
        setPrograms([program, ...programs])
        setSelectedProgram(program)
        setView(navigateTo)
    }

    const handleProgramSelect = (program: Program) => {
        setSelectedProgram(program)
        setView('overview')
    }

    const handleProgramDelete = (programId: string) => {
        const remainingPrograms = programs.filter(p => p.id !== programId)
        setPrograms(remainingPrograms)
        setSelectedProgram(remainingPrograms.length > 0 ? remainingPrograms[0] : null)
    }

    const handleNavigate = (newView: View) => {
        setView(newView)
    }

    if (loading) {
        return (
            <div className="loading loading-fullscreen">
                <div className="spinner"></div>
            </div>
        )
    }

    const renderContent = () => {
        if (view === 'new-program') {
            return (
                <CreateProgram
                    onCreated={handleProgramCreated}
                    onCancel={() => setView('overview')}
                />
            )
        }

        if (!selectedProgram) {
            return (
                <CreateProgram
                    onCreated={handleProgramCreated}
                    onCancel={() => {}}
                />
            )
        }

        switch (view) {
            case 'study':
                return <StudyInterface program={selectedProgram} onUpdate={loadPrograms} initialView="chat" />
            case 'flashcards':
                return <StudyInterface program={selectedProgram} onUpdate={loadPrograms} initialView="flashcards" />
            case 'documents':
                return (
                    <DocumentManager
                        program={selectedProgram}
                        onUpdate={loadPrograms}
                    />
                )
            case 'connectors':
                return (
                    <ConnectorManager
                        program={selectedProgram}
                        onUpdate={loadPrograms}
                    />
                )
            default:
                return (
                    <Dashboard
                        program={selectedProgram}
                        onProgramUpdate={loadPrograms}
                        onProgramDelete={handleProgramDelete}
                        onNavigate={handleNavigate}
                    />
                )
        }
    }

    return (
        <div className="app-layout">
            <Sidebar
                programs={programs}
                selectedProgram={selectedProgram}
                onSelectProgram={handleProgramSelect}
                currentView={view}
                onNavigate={handleNavigate}
            />

            <main className="main-content">
                {renderContent()}
            </main>
        </div>
    )
}
