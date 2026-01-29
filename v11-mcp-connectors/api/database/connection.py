"""Database connection and session management for v11."""

import os
from contextlib import contextmanager, asynccontextmanager
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session

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
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)
print(f"Using PostgreSQL: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'localhost'}")

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _run_v11_migrations(engine):
    """Add v11 columns to existing tables.

    SQLAlchemy's create_all() creates new tables but does NOT alter existing ones.
    This runs ALTER TABLE for any missing v11 columns on the documents table.
    """
    inspector = inspect(engine)

    # Check if documents table exists (it should from v10)
    if "documents" not in inspector.get_table_names():
        return  # create_all() will handle creating it fresh

    existing_columns = {col["name"] for col in inspector.get_columns("documents")}

    migrations = []
    if "source_type" not in existing_columns:
        migrations.append(
            "ALTER TABLE documents ADD COLUMN source_type VARCHAR(20) DEFAULT 'upload'"
        )
    if "source_url" not in existing_columns:
        migrations.append(
            "ALTER TABLE documents ADD COLUMN source_url VARCHAR(2000)"
        )
    if "connector_id" not in existing_columns:
        migrations.append(
            "ALTER TABLE documents ADD COLUMN connector_id VARCHAR(36) "
            "REFERENCES connector_configs(id)"
        )

    if migrations:
        with engine.begin() as conn:
            for sql in migrations:
                conn.execute(text(sql))
                print(f"  Migration: {sql.split('ADD COLUMN ')[1]}")
        print(f"Applied {len(migrations)} v11 migration(s) to documents table")
    else:
        print("v11 schema is up to date")


def init_database():
    """Create all tables if they don't exist and ensure default user exists."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created/verified")

    # Apply v11 column migrations for existing tables
    _run_v11_migrations(engine)

    # Ensure default user exists (for development)
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


def get_or_create_user(db: Session, user_id: str) -> User:
    """Get a user by ID, creating if they don't exist.

    In v10, we prepare for multi-user by scoping data to user_id,
    but actual authentication is deferred to Chapter 12.
    """
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        user = User(id=user_id, preferences={})
        db.add(user)
        db.commit()
        db.refresh(user)
    return user
