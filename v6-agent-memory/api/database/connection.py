"""Database connection and session management."""

import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base, User

# PostgreSQL connection (required)
POSTGRES_URL = os.environ.get("POSTGRES_URL")

if not POSTGRES_URL:
    raise RuntimeError(
        "POSTGRES_URL environment variable is required. "
        "Set it in your .env file, e.g.: POSTGRES_URL=postgresql://localhost/studybuddy"
    )

# Fix URL scheme for SQLAlchemy 2.0 (Vercel uses postgres://)
DATABASE_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,
    max_overflow=10,
)
print(f"Using PostgreSQL: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'localhost'}")

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_database():
    """Create all tables if they don't exist and ensure default user exists."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized")

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
