"""Database connection and session management."""

import os
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base, User

# Environment detection
IS_VERCEL = os.environ.get("VERCEL") == "1"
TURSO_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

# Database configuration priority:
# 1. Turso (if credentials provided) - works everywhere
# 2. Local SQLite file (for development)
# 3. In-memory SQLite (Vercel fallback without Turso)

if TURSO_URL and TURSO_TOKEN:
    # Turso: use libSQL with remote database
    DATABASE_URL = f"sqlite+libsql://{TURSO_URL}?secure=true"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"auth_token": TURSO_TOKEN},
        echo=False,
    )
    DB_PATH = "turso"
elif IS_VERCEL:
    # Vercel without Turso: in-memory (limited - data won't persist)
    DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    DB_PATH = ":memory:"
else:
    # Local development: SQLite file
    DB_PATH = Path(__file__).parent.parent.parent / "studybuddy.db"
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
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
