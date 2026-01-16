import { Link, useLocation } from 'react-router-dom'

function Sidebar({ programs, selectedProgram, onSelectProgram, onProgramCreated }) {
  const location = useLocation()

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>
          StudyBuddy
        </h1>
        <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
          v10 - Learn Anything
        </span>
      </div>

      <nav style={{ marginTop: '1.5rem' }}>
        <Link
          to="/"
          className={`tab ${location.pathname === '/' ? 'active' : ''}`}
          style={{
            display: 'block',
            padding: '0.5rem',
            borderRadius: 'var(--radius-md)',
            marginBottom: '0.25rem',
          }}
        >
          Overview
        </Link>
        <Link
          to="/study"
          className={`tab ${location.pathname === '/study' ? 'active' : ''}`}
          style={{
            display: 'block',
            padding: '0.5rem',
            borderRadius: 'var(--radius-md)',
            marginBottom: '0.25rem',
          }}
        >
          Study
        </Link>
        <Link
          to="/documents"
          className={`tab ${location.pathname === '/documents' ? 'active' : ''}`}
          style={{
            display: 'block',
            padding: '0.5rem',
            borderRadius: 'var(--radius-md)',
            marginBottom: '0.25rem',
          }}
        >
          Documents
        </Link>
      </nav>

      <div style={{ marginTop: '2rem' }}>
        <div className="flex justify-between items-center">
          <h3 style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
            Programs
          </h3>
          <Link
            to="/new-program"
            className="btn-secondary"
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
          >
            + New
          </Link>
        </div>

        <ul className="program-list">
          {programs.length === 0 ? (
            <li className="empty-state" style={{ padding: '1rem' }}>
              No programs yet
            </li>
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
                  {program.document_count} docs · {program.flashcard_count} cards
                </div>
              </li>
            ))
          )}
        </ul>
      </div>

      <div style={{ marginTop: 'auto', paddingTop: '1rem' }}>
        <a
          href="/api/health"
          target="_blank"
          style={{
            fontSize: '0.75rem',
            color: 'var(--color-text-muted)',
          }}
        >
          API Health Check
        </a>
      </div>
    </aside>
  )
}

export default Sidebar
