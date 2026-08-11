"""
NeuroPulse AI — Database Setup
SQLite for development (no PostgreSQL install needed).
Easy switch to PostgreSQL for production.
"""

from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession
)
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.app.config import settings

# ── Engine ─────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
    if "sqlite" in settings.DATABASE_URL else {}
)

# ── Session ────────────────────────────────────────────────────────────────────
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# ── Base ───────────────────────────────────────────────────────────────────────
Base = declarative_base()


async def create_tables():
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency for database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()