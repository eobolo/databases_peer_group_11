# api/database.py
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException

DATABASE_URL = "postgresql://neondb_owner:npg_S8dxteJ7yYmo@ep-round-river-a59jaxkv-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

def get_db_connection():
    """Establishes a connection to the Neon database."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def resolve_category(cur, table_info: dict, value: str | None) -> int | None:
    """Resolves a categorical value to its corresponding ID."""
    if value is None:
        return None
    table = table_info["table"]
    name_column = table_info["name_column"]
    id_column = table_info["id_column"]  # Use explicit id_column from table_info
    cur.execute(f"SELECT {id_column} FROM {table} WHERE {name_column} = %s", (value,))
    result = cur.fetchone()
    if result is None:
        raise HTTPException(status_code=400, detail=f"Invalid {name_column}: {value}")
    return result[id_column]

def get_unique_values(cur, column: str, table_map: dict) -> list[str]:
    """Fetches unique values for a categorical column."""
    if column not in table_map:
        raise HTTPException(status_code=400, detail=f"Column '{column}' is not categorical")
    table_info = table_map[column]
    table = table_info["table"]
    name_column = table_info["name_column"]
    cur.execute(f"SELECT DISTINCT {name_column} FROM {table} ORDER BY {name_column}")
    results = cur.fetchall()
    return [row[name_column] for row in results]