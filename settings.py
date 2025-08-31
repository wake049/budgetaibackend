import os
from urllib.parse import quote_plus

def build_db_url():
    url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
    if url:
        return url  # if you ever set it explicitly

    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")

    if not all([user, pwd, host, name]):
        return None

    return f"postgresql+psycopg2://{user}:{quote_plus(pwd)}@{host}:{port}/{name}"

DATABASE_URL = build_db_url()