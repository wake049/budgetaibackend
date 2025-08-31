from .session import SessionLocal

def get_db():
    """
    Dependency to get a database session.
    Yields a database session and ensures it is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()