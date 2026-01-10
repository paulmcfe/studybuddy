"""Database connection and session management."""

from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base, User

# Database file location (same directory as the v6 package root)
DB_PATH = Path(__file__).parent.parent.parent / "studybuddy.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create engine with SQLite-specific settings
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for FastAPI
    echo=False,  # Set True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """Create all tables if they don't exist and ensure default user exists."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at {DB_PATH}")

    # Ensure default user exists
    db = SessionLocal()
    try:
        default_user = db.query(User).filter_by(id="default").first()
        if not default_user:
            db.add(User(id="default", preferences={}))
            db.commit()
            print("Created default user")
    finally:
        db.close()


@contextmanager
def get_db():
    """Context manager for database sessions.

    Usage:
        with get_db() as db:
            user = db.query(User).filter_by(id="default").first()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_dependency():
    """FastAPI dependency for database sessions.

    Usage in FastAPI:
        @app.get("/api/something")
        def something(db: Session = Depends(get_db_dependency)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
