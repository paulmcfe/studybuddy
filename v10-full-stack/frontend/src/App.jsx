import { useState, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Dashboard from './components/Dashboard'
import StudyInterface from './components/StudyInterface'
import DocumentManager from './components/DocumentManager'
import CreateProgram from './components/CreateProgram'

function App() {
  const [programs, setPrograms] = useState([])
  const [selectedProgram, setSelectedProgram] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadPrograms()
  }, [])

  const loadPrograms = async () => {
    try {
      const response = await fetch('/api/programs')
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

  const handleProgramCreated = (program) => {
    setPrograms([program, ...programs])
    setSelectedProgram(program)
  }

  const handleProgramSelect = (program) => {
    setSelectedProgram(program)
  }

  const handleProgramDelete = (programId) => {
    const remainingPrograms = programs.filter(p => p.id !== programId)
    setPrograms(remainingPrograms)
    setSelectedProgram(remainingPrograms.length > 0 ? remainingPrograms[0] : null)
  }

  if (loading) {
    return (
      <div className="loading" style={{ height: '100vh' }}>
        <div className="spinner"></div>
      </div>
    )
  }

  return (
    <div className="app-layout">
      <Sidebar
        programs={programs}
        selectedProgram={selectedProgram}
        onSelectProgram={handleProgramSelect}
        onProgramCreated={handleProgramCreated}
      />

      <main className="main-content">
        <Routes>
          <Route
            path="/"
            element={
              selectedProgram ? (
                <Dashboard
                  program={selectedProgram}
                  onProgramUpdate={loadPrograms}
                  onProgramDelete={handleProgramDelete}
                />
              ) : (
                <CreateProgram onCreated={handleProgramCreated} />
              )
            }
          />
          <Route
            path="/study"
            element={
              selectedProgram ? (
                <StudyInterface program={selectedProgram} />
              ) : (
                <Navigate to="/" />
              )
            }
          />
          <Route
            path="/documents"
            element={
              selectedProgram ? (
                <DocumentManager
                  program={selectedProgram}
                  onUpdate={loadPrograms}
                />
              ) : (
                <Navigate to="/" />
              )
            }
          />
          <Route
            path="/new-program"
            element={<CreateProgram onCreated={handleProgramCreated} />}
          />
        </Routes>
      </main>
    </div>
  )
}

export default App
